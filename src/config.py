"""Centralized, environment-driven configuration for the supply chain optimizer.

All runtime configuration is loaded from environment variables or a local
``.env`` file via ``pydantic-settings``. Nothing (tracking URIs, ports,
scenario names) is hardcoded into business logic, which keeps the codebase
decoupled and deployment-ready for hyperscaler environments such as SAP AI Core.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root resolved relative to this file (src/config.py -> repo root).
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"


class Settings(BaseSettings):
    """Strongly-typed application settings sourced from the environment.

    Attributes:
        random_seed: Global seed applied across numpy/torch for reproducibility.
        n_records: Number of synthetic ERP records to synthesize.
        stockout_rate: Target positive-class ratio for the imbalanced dataset.
        train_frac: Fraction of the chronological timeline used for training.
        val_frac: Fraction of the chronological timeline used for validation.
        mlflow_tracking_uri: MLflow tracking backend URI.
        mlflow_experiment_name: Logical MLflow experiment grouping.
        sap_ai_core_scenario: Mocked SAP AI Core scenario name.
        sap_ai_core_executable: Mocked SAP AI Core executable name.
        sap_ai_core_resource_group: Mocked SAP AI Core resource group.
        llm_provider: Which LLM backend the agent uses ("mock" for offline).
        port: Port the FastAPI inference service binds to.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Reproducibility
    random_seed: int = 42

    # Data synthesis
    n_records: int = 50_000
    stockout_rate: float = 0.02
    train_frac: float = 0.70
    val_frac: float = 0.15  # test_frac is implied as the remainder (0.15)

    # MLflow. SQLite backend keeps runs local/offline while remaining
    # compatible with modern MLflow (the file store is deprecated).
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "Supply_Chain_Imbalanced_Predictor"

    # SAP AI Core (mocked)
    sap_ai_core_scenario: str = "supply-chain-risk-optimizer"
    sap_ai_core_executable: str = "stockout-predictor"
    sap_ai_core_resource_group: str = "default"

    # Agent
    llm_provider: str = "mock"

    # Deployment
    port: int = 8080


# Single shared settings instance imported across the codebase.
settings = Settings()
