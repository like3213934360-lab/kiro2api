# -*- coding: utf-8 -*-

"""
Tests for kiro/account_manager.py - Unified Account System.

Tests the AccountManager class that manages multiple Kiro accounts with:
- Lazy initialization
- Sticky behavior (prefer successful account)
- Circuit breaker with exponential backoff
- TTL-based model cache refresh
- State persistence
"""

import asyncio
import json
import pytest
import time
import httpx
from fastapi import HTTPException
from pathlib import Path
from typing import List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from kiro.account_manager import (
    Account,
    AccountStats,
    ModelAccountList,
    AccountManager,
    _fetch_model_catalog,
    _fallback_model_catalog,
    _format_duration
)
from kiro.account_errors import ErrorType
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.config import FALLBACK_MODELS
from kiro.model_resolver import ModelResolver


class TestAccountDataclass:
    """
    Tests for Account and AccountStats dataclasses.
    """
    
    def test_account_creation_with_defaults(self):
        """
        Test Account creation with default values.
        
        What it does: Verifies Account dataclass initialization
        Purpose: Ensure default values are set correctly
        """
        print("\n=== Test: Account creation with defaults ===")
        
        # Act
        account = Account(id="/test/path.json")
        
        # Assert
        print(f"Account ID: {account.id}")
        print(f"Auth manager: {account.auth_manager}")
        print(f"Failures: {account.failures}")
        print(f"Last failure time: {account.last_failure_time}")
        
        assert account.id == "/test/path.json"
        assert account.auth_manager is None
        assert account.model_cache is None
        assert account.model_resolver is None
        assert account.failures == 0
        assert account.last_failure_time == 0.0
        assert account.models_cached_at == 0.0
        assert isinstance(account.stats, AccountStats)
    
    def test_account_stats_initialization(self):
        """
        Test AccountStats initialization with zeros.
        
        What it does: Verifies AccountStats default values
        Purpose: Ensure statistics start at zero
        """
        print("\n=== Test: AccountStats initialization ===")
        
        # Act
        stats = AccountStats()
        
        # Assert
        print(f"Total requests: {stats.total_requests}")
        print(f"Successful requests: {stats.successful_requests}")
        print(f"Failed requests: {stats.failed_requests}")
        
        assert stats.total_requests == 0
        assert stats.successful_requests == 0
        assert stats.failed_requests == 0


class TestAccountManagerLoadCredentials:
    """
    Tests for AccountManager.load_credentials() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_credentials_json_type(self, tmp_path):
        """
        Test loading credentials with type=json.
        
        What it does: Loads single JSON credential file
        Purpose: Verify JSON type credential loading
        """
        print("\n=== Test: load_credentials with type=json ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        assert str(test_json.resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_sqlite_type(self, tmp_path, temp_sqlite_db):
        """
        Test loading credentials with type=sqlite.
        
        What it does: Loads SQLite database credential
        Purpose: Verify SQLite type credential loading
        """
        print("\n=== Test: load_credentials with type=sqlite ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "sqlite",
                "path": temp_sqlite_db,
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1
        assert str(Path(temp_sqlite_db).resolve()) in manager._accounts
    
    @pytest.mark.asyncio
    async def test_load_credentials_refresh_token_type(self, tmp_path):
        """
        Test loading credentials with type=refresh_token.
        
        What it does: Loads refresh token credential
        Purpose: Verify refresh_token type credential loading
        """
        print("\n=== Test: load_credentials with type=refresh_token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "refresh_token": "test_refresh_token_abc123",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "region": "us-east-1",
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        # Create state file to avoid errors
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps({"current_account_index": 0, "model_to_accounts": {}, "accounts": {}}))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        print(f"Account IDs: {list(manager._accounts.keys())}")
        
        assert len(manager._accounts) == 1
        # refresh_token type uses deterministic hash as ID
        account_id = list(manager._accounts.keys())[0]
        assert account_id.startswith("refresh_token_")
    
    @pytest.mark.asyncio
    async def test_load_credentials_folder_scanning(self, tmp_path):
        """
        Test folder scanning for credential files.
        
        What it does: Scans folder and loads all valid credential files
        Purpose: Verify folder scanning functionality
        """
        print("\n=== Test: load_credentials with folder scanning ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Create valid files
        file1 = folder / "account1.json"
        file1.write_text(json.dumps({
            "refreshToken": "token1",
            "accessToken": "access1",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        file2 = folder / "account2.json"
        file2.write_text(json.dumps({
            "refreshToken": "token2",
            "accessToken": "access2",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_invalid_files(self, tmp_path):
        """
        Test that invalid files are skipped with WARNING.
        
        What it does: Loads folder with invalid files
        Purpose: Verify invalid files are skipped gracefully
        """
        print("\n=== Test: load_credentials skips invalid files ===")
        
        # Arrange
        folder = tmp_path / "accounts"
        folder.mkdir()
        
        # Valid file
        valid_file = folder / "valid.json"
        valid_file.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        # Invalid JSON
        invalid_file = folder / "invalid.json"
        invalid_file.write_text("not a valid json {{{")
        
        # Non-JSON file
        text_file = folder / "readme.txt"
        text_file.write_text("This is not a credential file")
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(folder),
                "enabled": True
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 1  # Only valid file loaded
    
    @pytest.mark.asyncio
    async def test_load_credentials_skip_disabled(self, tmp_path):
        """
        Test that entries with enabled=false are skipped.
        
        What it does: Loads credentials with disabled entry
        Purpose: Verify enabled flag is respected
        """
        print("\n=== Test: load_credentials skips disabled entries ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "token",
            "accessToken": "access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "path": str(test_json),
                "enabled": False  # Disabled
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_type(self, tmp_path):
        """
        Test that entries without type are skipped.
        
        What it does: Loads credentials with missing type field
        Purpose: Verify type validation
        """
        print("\n=== Test: load_credentials skips entries without type ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "path": "/some/path.json",
                "enabled": True
                # Missing "type" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_path(self, tmp_path):
        """
        Test that json/sqlite entries without path are skipped.
        
        What it does: Loads credentials with missing path field
        Purpose: Verify path validation for json/sqlite types
        """
        print("\n=== Test: load_credentials skips json/sqlite without path ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "json",
                "enabled": True
                # Missing "path" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_missing_refresh_token(self, tmp_path):
        """
        Test that refresh_token entries without refresh_token field are skipped.
        
        What it does: Loads credentials with missing refresh_token field
        Purpose: Verify refresh_token validation
        """
        print("\n=== Test: load_credentials skips refresh_token without token ===")
        
        # Arrange
        creds_file = tmp_path / "credentials.json"
        credentials = [
            {
                "type": "refresh_token",
                "profile_arn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
                "enabled": True
                # Missing "refresh_token" field
            }
        ]
        creds_file.write_text(json.dumps(credentials))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0
    
    @pytest.mark.asyncio
    async def test_load_credentials_file_not_found(self, tmp_path):
        """
        Test handling of non-existent credentials.json.
        
        What it does: Attempts to load non-existent file
        Purpose: Verify graceful handling of missing file
        """
        print("\n=== Test: load_credentials with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "nonexistent.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act
        await manager.load_credentials()
        
        # Assert
        print(f"Loaded accounts: {len(manager._accounts)}")
        
        assert len(manager._accounts) == 0


class TestAccountManagerLoadState:
    """
    Tests for AccountManager.load_state() method.
    """
    
    @pytest.mark.asyncio
    async def test_load_state_success(self, tmp_path, sample_state_with_data):
        """
        Test loading existing state.json.
        
        What it does: Loads state from file
        Purpose: Verify state restoration
        """
        print("\n=== Test: load_state success ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(sample_state_with_data))
        
        # Create accounts first
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) > 0
    
    @pytest.mark.asyncio
    async def test_load_state_restore_current_account_index(self, tmp_path):
        """
        Test restoration of global current_account_index.
        
        What it does: Restores sticky index from state
        Purpose: Verify global sticky behavior persistence
        """
        print("\n=== Test: load_state restores current_account_index ===")
        
        # Arrange
        state_data = {
            "current_account_index": 2,
            "model_to_accounts": {},
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Current account index: {manager._current_account_index}")
        
        assert manager._current_account_index == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_model_to_accounts(self, tmp_path):
        """
        Test restoration of model_to_accounts mapping.
        
        What it does: Restores model mappings from state
        Purpose: Verify model-to-account mapping persistence
        """
        print("\n=== Test: load_state restores model_to_accounts ===")
        
        # Arrange
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {
                "claude-opus-4.5": {
                    "accounts": ["/test/account1.json", "/test/account2.json"]
                }
            },
            "accounts": {}
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {manager._model_to_accounts}")
        
        assert "claude-opus-4.5" in manager._model_to_accounts
        assert len(manager._model_to_accounts["claude-opus-4.5"].accounts) == 2
    
    @pytest.mark.asyncio
    async def test_load_state_restore_account_runtime_state(self, tmp_path):
        """
        Test restoration of account runtime state (failures, stats, etc).
        
        What it does: Restores account state from file
        Purpose: Verify runtime state persistence
        """
        print("\n=== Test: load_state restores account runtime state ===")
        
        # Arrange
        # Create account first to get correct resolved path
        test_json = tmp_path / "account.json"
        test_json.write_text(json.dumps({"refreshToken": "token"}))
        account_id = str(test_json.resolve())
        
        state_data = {
            "current_account_index": 0,
            "model_to_accounts": {},
            "accounts": {
                account_id: {
                    "failures": 3,
                    "last_failure_time": 1704110400.0,
                    "models_cached_at": 1704106800.0,
                    "stats": {
                        "total_requests": 100,
                        "successful_requests": 97,
                        "failed_requests": 3
                    }
                }
            }
        }
        
        state_file = tmp_path / "state.json"
        state_file.write_text(json.dumps(state_data))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(state_file)
        )
        
        await manager.load_credentials()
        
        # Act
        await manager.load_state()
        
        # Assert
        account = manager._accounts[account_id]
        print(f"Account failures: {account.failures}")
        print(f"Account stats: {account.stats}")
        
        assert account.failures == 3
        assert account.last_failure_time == 1704110400.0
        assert account.models_cached_at == 1704106800.0
        assert account.stats.total_requests == 100
    
    @pytest.mark.asyncio
    async def test_load_state_file_not_found(self, tmp_path):
        """
        Test handling of non-existent state.json (empty state).
        
        What it does: Attempts to load non-existent state file
        Purpose: Verify graceful handling with empty state
        """
        print("\n=== Test: load_state with missing file ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "nonexistent.json")
        )
        
        # Act
        await manager.load_state()
        
        # Assert
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        print(f"Current account index: {manager._current_account_index}")
        
        assert len(manager._model_to_accounts) == 0
        assert manager._current_account_index == 0
    
    @pytest.mark.asyncio
    async def test_load_state_corrupted_json(self, tmp_path):
        """
        Test handling of corrupted state.json.
        
        What it does: Attempts to load invalid JSON
        Purpose: Verify error handling for corrupted state
        """
        print("\n=== Test: load_state with corrupted JSON ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        state_file.write_text("not a valid json {{{")
        
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager.load_state()
        
        # Assert - should handle gracefully
        print(f"Model mappings: {len(manager._model_to_accounts)}")
        
        assert len(manager._model_to_accounts) == 0



class TestAccountManagerInitializeAccount:
    """
    Tests for AccountManager._initialize_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_initialize_account_json_success(self, tmp_path, mock_list_models_response):
        """
        Test successful account initialization with type=json.
        
        What it does: Initializes account with JSON credentials
        Purpose: Verify complete initialization flow
        """
        print("\n=== Test: initialize_account with JSON ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z",
            "profileArn": "arn:aws:codewhisperer:us-east-1:123456789:profile/test",
            "region": "us-east-1"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client for ListAvailableModels
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True
        assert manager._accounts[account_id].auth_manager is not None
        assert manager._accounts[account_id].model_cache is not None
        assert manager._accounts[account_id].model_resolver is not None
    
    @pytest.mark.asyncio
    async def test_initialize_account_fetch_models_fallback(self, tmp_path):
        """
        Test fallback to FALLBACK_MODELS when API fails.
        
        What it does: Initializes account when ListAvailableModels fails
        Purpose: Verify fallback mechanism
        """
        print("\n=== Test: initialize_account with fallback models ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Mock HTTP client to fail with a realistic network error
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_client.request_with_retry = AsyncMock(
                side_effect=httpx.ConnectError("Network error")
            )
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            # Act
            success = await manager._initialize_account(account_id)
        
        # Assert
        print(f"Initialization success: {success}")
        assert success is True  # Should succeed with fallback
        assert manager._accounts[account_id].model_cache is not None

        print("Verification: built-in catalog was loaded instead of an empty cache...")
        cached_models = manager._accounts[account_id].model_cache.get_all_model_ids()
        print(f"Cached models: {len(cached_models)}")
        assert len(cached_models) == len(FALLBACK_MODELS)
        assert "auto" in cached_models


class TestAccountManagerGetNextAccount:
    """
    Tests for AccountManager.get_next_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_next_account_single_bypass_circuit_breaker(self, tmp_path, mock_list_models_response):
        """
        Test that single account bypasses Circuit Breaker.
        
        What it does: Gets account when only one exists
        Purpose: Verify single account always returns (no cooldown)
        """
        print("\n=== Test: get_next_account single account bypasses Circuit Breaker ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures (should be ignored for single account)
        manager._accounts[account_id].failures = 10
        manager._accounts[account_id].last_failure_time = time.time()
        
        # Act
        account = await manager.get_next_account("claude-opus-4.5")
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None  # Single account always returns


class TestAccountManagerReportSuccess:
    """
    Tests for AccountManager.report_success() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_success_reset_failures(self, tmp_path, mock_list_models_response):
        """
        Test that report_success resets failures to 0.
        
        What it does: Reports success after failures
        Purpose: Verify failure counter reset
        """
        print("\n=== Test: report_success resets failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Set failures
        manager._accounts[account_id].failures = 5
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        print(f"Failures after success: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0
    
    @pytest.mark.asyncio
    async def test_report_success_update_stats(self, tmp_path, mock_list_models_response):
        """
        Test that report_success updates statistics.
        
        What it does: Reports success and checks stats
        Purpose: Verify statistics tracking
        """
        print("\n=== Test: report_success updates stats ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_success(account_id, "claude-opus-4.5")
        
        # Assert
        stats = manager._accounts[account_id].stats
        print(f"Stats: total={stats.total_requests}, successful={stats.successful_requests}")
        assert stats.total_requests == 1
        assert stats.successful_requests == 1


class TestAccountManagerReportFailure:
    """
    Tests for AccountManager.report_failure() method.
    """
    
    @pytest.mark.asyncio
    async def test_report_failure_recoverable_increment_failures(self, tmp_path, mock_list_models_response):
        """
        Test that RECOVERABLE errors increment failures.
        
        What it does: Reports RECOVERABLE failure
        Purpose: Verify failure counter increment
        """
        print("\n=== Test: report_failure RECOVERABLE increments failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.RECOVERABLE, 429, None
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 1
    
    @pytest.mark.asyncio
    async def test_report_failure_fatal_no_increment(self, tmp_path, mock_list_models_response):
        """
        Test that FATAL errors do NOT increment failures.
        
        What it does: Reports FATAL failure
        Purpose: Verify failures not incremented for request errors
        """
        print("\n=== Test: report_failure FATAL does not increment failures ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        await manager.report_failure(
            account_id, "claude-opus-4.5",
            ErrorType.FATAL, 400, "CONTENT_LENGTH_EXCEEDS_THRESHOLD"
        )
        
        # Assert
        print(f"Failures: {manager._accounts[account_id].failures}")
        assert manager._accounts[account_id].failures == 0  # Not incremented


class TestAccountManagerSaveState:
    """
    Tests for AccountManager._save_state() and save_state_periodically().
    """
    
    @pytest.mark.asyncio
    async def test_save_state_atomic_write(self, tmp_path):
        """
        Test atomic state saving via tmp file.
        
        What it does: Saves state and checks tmp file usage
        Purpose: Verify atomic write pattern
        """
        print("\n=== Test: save_state atomic write ===")
        
        # Arrange
        state_file = tmp_path / "state.json"
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(state_file)
        )
        
        # Act
        await manager._save_state()
        
        # Assert
        print(f"State file exists: {state_file.exists()}")
        assert state_file.exists()
        
        # Verify tmp file was cleaned up
        tmp_file = tmp_path / "state.json.tmp"
        print(f"Tmp file exists: {tmp_file.exists()}")
        assert not tmp_file.exists()


class TestAccountManagerGetFirstAccount:
    """
    Tests for AccountManager.get_first_account() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_first_account_success(self, tmp_path, mock_list_models_response):
        """
        Test getting first initialized account.
        
        What it does: Gets first account for legacy mode
        Purpose: Verify legacy mode support
        """
        print("\n=== Test: get_first_account success ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        account = manager.get_first_account()
        
        # Assert
        print(f"Got account: {account is not None}")
        assert account is not None
        assert account.auth_manager is not None
    
    def test_get_first_account_no_initialized(self, tmp_path):
        """
        Test RuntimeError when no initialized accounts.
        
        What it does: Attempts to get account when none initialized
        Purpose: Verify error handling
        """
        print("\n=== Test: get_first_account with no initialized accounts ===")
        
        # Arrange
        manager = AccountManager(
            credentials_file=str(tmp_path / "creds.json"),
            state_file=str(tmp_path / "state.json")
        )
        
        # Act & Assert
        with pytest.raises(RuntimeError, match="No initialized accounts available"):
            manager.get_first_account()


class TestAccountManagerGetAllAvailableModels:
    """
    Tests for AccountManager.get_all_available_models() method.
    """
    
    @pytest.mark.asyncio
    async def test_get_all_available_models_collect_from_all(self, tmp_path, mock_list_models_response):
        """
        Test collecting unique models from all accounts.
        
        What it does: Gets models from multiple accounts
        Purpose: Verify model aggregation for /v1/models endpoint
        """
        print("\n=== Test: get_all_available_models collects from all ===")
        
        # Arrange
        test_json = tmp_path / "test.json"
        test_json.write_text(json.dumps({
            "refreshToken": "test_token",
            "accessToken": "test_access",
            "expiresAt": "2099-01-01T00:00:00.000Z"
        }))
        
        creds_file = tmp_path / "credentials.json"
        creds_file.write_text(json.dumps([
            {"type": "json", "path": str(test_json), "enabled": True}
        ]))
        
        manager = AccountManager(
            credentials_file=str(creds_file),
            state_file=str(tmp_path / "state.json")
        )
        
        await manager.load_credentials()
        account_id = str(test_json.resolve())
        
        # Initialize account
        with patch('kiro.account_manager.KiroHttpClient') as mock_http_class:
            mock_client = AsyncMock()
            mock_response = Mock()  # Response is not async
            mock_response.status_code = 200
            mock_response.json.return_value = mock_list_models_response
            mock_client.request_with_retry = AsyncMock(return_value=mock_response)
            mock_client.close = AsyncMock()
            mock_http_class.return_value = mock_client
            
            await manager._initialize_account(account_id)
        
        # Act
        models = manager.get_all_available_models()
        
        # Assert
        print(f"Available models: {len(models)}")
        assert len(models) > 0
        assert isinstance(models, list)
        assert all(isinstance(m, str) for m in models)


class TestFormatDuration:
    """
    Tests for _format_duration() helper function.
    """
    
    def test_format_duration_seconds(self):
        """Test formatting seconds."""
        assert _format_duration(30) == "30s"
        assert _format_duration(59) == "59s"
    
    def test_format_duration_minutes(self):
        """Test formatting minutes."""
        assert _format_duration(60) == "1m"
        assert _format_duration(300) == "5m"
        assert _format_duration(3599) == "59m"
    
    def test_format_duration_hours(self):
        """Test formatting hours."""
        assert _format_duration(3600) == "1h"
        assert _format_duration(7200) == "2h"
        assert _format_duration(86399) == "23h"
    
    def test_format_duration_days(self):
        """Test formatting days."""
        assert _format_duration(86400) == "1d"
        assert _format_duration(172800) == "2d"


# =============================================================================
# Tests for the model catalog fetch (_fetch_model_catalog)
# =============================================================================

def _make_catalog_auth(
    models_host: str = "https://catalog.example",
    q_host: str = "https://q.example",
    profile_arn: Optional[str] = None
) -> MagicMock:
    """Build a minimal auth-manager double exposing the host properties."""
    auth = MagicMock()
    auth.models_host = models_host
    auth.q_host = q_host
    auth.profile_arn = profile_arn
    return auth


def _make_catalog_response(models: List, profile_arn: Optional[str] = None) -> MagicMock:
    """Build an httpx-Response-like double for /ListAvailableModels."""
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {"models": models, "profileArn": profile_arn}
    return response


class TestFetchModelCatalog:
    """
    Tests for _fetch_model_catalog().

    The runtime host returns 404 for /ListAvailableModels, so the helper must try
    the control-plane host first and fall back to the Q API host.
    """

    @pytest.mark.asyncio
    async def test_uses_catalog_host_first(self):
        """
        What it does: Verifies the catalog host is queried before the Q API host.
        Purpose: This is the fix for /v1/models returning a single model.
        """
        print("Setup: auth with distinct catalog/Q hosts...")
        auth = _make_catalog_auth()
        urls = []

        async def _fake_request(method, url, json_data, params, stream):
            urls.append(url)
            return _make_catalog_response([{"modelId": "claude-sonnet-4.5"}], "arn:aws:x")

        mock_client = AsyncMock()
        mock_client.request_with_retry = _fake_request
        mock_client.close = AsyncMock()

        print("Action: _fetch_model_catalog(auth)...")
        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client):
            models, profile_arn = await _fetch_model_catalog(auth)

        print(f"Received models: {models}, profileArn: {profile_arn}")
        assert models == [{"modelId": "claude-sonnet-4.5"}]
        assert profile_arn == "arn:aws:x"
        assert urls == ["https://catalog.example/ListAvailableModels"]
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_falls_back_to_q_host_when_catalog_host_fails(self):
        """
        What it does: Verifies failover to the Q API host on catalog-host errors.
        Purpose: Older/corporate deployments expose /ListAvailableModels elsewhere.
        """
        print("Setup: catalog host raises, Q API host answers...")
        auth = _make_catalog_auth()
        urls = []

        async def _fake_request(method, url, json_data, params, stream):
            urls.append(url)
            if url.startswith("https://catalog.example"):
                raise HTTPException(status_code=404, detail="Not Found")
            return _make_catalog_response([{"modelId": "auto"}])

        mock_client = AsyncMock()
        mock_client.request_with_retry = _fake_request
        mock_client.close = AsyncMock()

        print("Action: _fetch_model_catalog(auth)...")
        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client):
            models, profile_arn = await _fetch_model_catalog(auth)

        print(f"Attempted hosts: {urls}")
        assert models == [{"modelId": "auto"}]
        assert profile_arn is None
        assert urls == [
            "https://catalog.example/ListAvailableModels",
            "https://q.example/ListAvailableModels"
        ]

    @pytest.mark.asyncio
    async def test_raises_when_every_host_fails(self):
        """
        What it does: Verifies RuntimeError when no host answers.
        Purpose: Callers must be able to detect exhaustion and use the fallback list.
        """
        print("Setup: every host raises ConnectError...")
        auth = _make_catalog_auth()

        async def _fake_request(method, url, json_data, params, stream):
            raise httpx.ConnectError("dns failure")

        mock_client = AsyncMock()
        mock_client.request_with_retry = _fake_request
        mock_client.close = AsyncMock()

        print("Action: _fetch_model_catalog(auth) expecting RuntimeError...")
        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                await _fetch_model_catalog(auth)

        print(f"Raised: {exc_info.value}")
        print("Verification: both hosts are reported in the message...")
        assert "catalog.example" in str(exc_info.value)
        assert "q.example" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_raises_on_empty_catalog(self):
        """
        What it does: Verifies an empty "models" array is treated as a failure.
        Purpose: An empty catalog means "unknown", not "no models available".
        """
        print("Setup: every host answers with an empty catalog...")
        auth = _make_catalog_auth()

        async def _fake_request(method, url, json_data, params, stream):
            return _make_catalog_response([])

        mock_client = AsyncMock()
        mock_client.request_with_retry = _fake_request
        mock_client.close = AsyncMock()

        print("Action: _fetch_model_catalog(auth) expecting RuntimeError...")
        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client):
            with pytest.raises(RuntimeError) as exc_info:
                await _fetch_model_catalog(auth)

        print(f"Raised: {exc_info.value}")
        assert "empty model list" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_duplicate_hosts_are_queried_once(self):
        """
        What it does: Verifies the same host is not retried twice.
        Purpose: Some deployments use one host for both roles; avoid double traffic.
        """
        print("Setup: catalog and Q hosts are identical...")
        auth = _make_catalog_auth(
            models_host="https://same.example",
            q_host="https://same.example"
        )
        urls = []

        async def _fake_request(method, url, json_data, params, stream):
            urls.append(url)
            raise HTTPException(status_code=404, detail="Not Found")

        mock_client = AsyncMock()
        mock_client.request_with_retry = _fake_request
        mock_client.close = AsyncMock()

        print("Action: _fetch_model_catalog(auth) expecting RuntimeError...")
        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client):
            with pytest.raises(RuntimeError):
                await _fetch_model_catalog(auth)

        print(f"Attempted hosts: {urls}")
        assert urls == ["https://same.example/ListAvailableModels"]

    @pytest.mark.asyncio
    async def test_profile_arn_is_sent_only_when_present(self):
        """
        What it does: Verifies profileArn is forwarded as a query parameter.
        Purpose: Kiro requires profileArn for accounts that have one, and rejects
                 it being absent for those accounts.
        """
        print("Setup: auth with a profile ARN...")
        auth = _make_catalog_auth(profile_arn="arn:aws:codewhisperer:us-east-1:1:profile/X")
        captured_params = {}

        async def _fake_request(method, url, json_data, params, stream):
            captured_params.update(params)
            return _make_catalog_response([{"modelId": "auto"}])

        mock_client = AsyncMock()
        mock_client.request_with_retry = _fake_request
        mock_client.close = AsyncMock()

        print("Action: _fetch_model_catalog(auth)...")
        with patch("kiro.account_manager.KiroHttpClient", return_value=mock_client):
            await _fetch_model_catalog(auth)

        print(f"Params sent: {captured_params}")
        assert captured_params["origin"] == "AI_EDITOR"
        assert captured_params["profileArn"] == "arn:aws:codewhisperer:us-east-1:1:profile/X"


class TestFallbackModelCatalog:
    """Tests for _fallback_model_catalog()."""

    def test_returns_defensive_copy(self):
        """
        What it does: Verifies mutating the returned list does not touch FALLBACK_MODELS.
        Purpose: A per-account cache must never corrupt the module-level catalog.
        """
        print("Action: _fallback_model_catalog() twice...")
        first = _fallback_model_catalog()
        second = _fallback_model_catalog()

        print("Mutation: appending to the first copy...")
        first.append({"modelId": "bogus"})

        print(f"Comparing lengths: {len(first)} vs {len(FALLBACK_MODELS)}")
        assert len(first) == len(FALLBACK_MODELS) + 1
        assert len(second) == len(FALLBACK_MODELS)

    def test_contains_known_current_models(self):
        """
        What it does: Verifies the catalog carries the models users actually get.
        Purpose: A stale list silently hides new models from every client.
        """
        print("Action: _fallback_model_catalog()...")
        model_ids = {model["modelId"] for model in _fallback_model_catalog()}

        print(f"Catalog: {sorted(model_ids)}")
        for expected in (
            "auto",
            "claude-opus-5.5",
            "claude-sonnet-5.5",
            "claude-haiku-5.5",
            "claude-sonnet-4.6",
            "kimi-k3",
            "qwen3-coder-next",
        ):
            print(f"Verification: {expected} present...")
            assert expected in model_ids
