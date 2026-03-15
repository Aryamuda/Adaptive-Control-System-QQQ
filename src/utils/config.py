"""
Configuration loader for the Adaptive Control System.
Loads parameters from config.yaml with validation and defaults.
"""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class PipelineConfig:
    target_minutes: int = 5
    ttl_minutes: int = 15


@dataclass
class FeaturesConfig:
    buffer_capacity: int = 5
    expected_strikes: int = 120
    expected_columns: List[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    eval_metric: str = "auc"


@dataclass
class ControlConfig:
    kp: float = 0.5
    ki: float = 0.1
    kd: float = 0.0
    alpha: float = 0.05
    min_weight: float = 0.1
    max_weight: float = 2.0


@dataclass
class LabelerConfig:
    target_minutes: int = 5
    threshold: float = 0.0005


@dataclass
class FeedbackConfig:
    max_null_greek_ratio: float = 0.1
    max_zero_iv_ratio: float = 0.1
    high_vol_threshold: float = 0.30


@dataclass
class RegimeConfig:
    open_time: str = "10:30:00"
    close_time: str = "15:00:00"


@dataclass
class LoggingConfig:
    log_file: str = "signals.log"
    level: str = "INFO"


@dataclass
class AppConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    labeler: LabelerConfig = field(default_factory=LabelerConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    regime: RegimeConfig = field(default_factory=RegimeConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def load_config(config_path: str = "config.yaml") -> AppConfig:
    """
    Load configuration from YAML file with validation.
    
    Args:
        config_path: Path to the config YAML file
        
    Returns:
        AppConfig instance with all configuration values
    """
    if not os.path.exists(config_path):
        return _get_default_config()
    
    with open(config_path, 'r') as f:
        raw = yaml.safe_load(f)
    
    if raw is None:
        return _get_default_config()
    
    # Build nested config objects
    config = AppConfig()
    
    if 'pipeline' in raw:
        config.pipeline = PipelineConfig(**raw['pipeline'])
    
    if 'features' in raw:
        config.features = FeaturesConfig(**raw['features'])
    
    if 'model' in raw:
        config.model = ModelConfig(**raw['model'])
    
    if 'control' in raw:
        config.control = ControlConfig(**raw['control'])
    
    if 'labeler' in raw:
        config.labeler = LabelerConfig(**raw['labeler'])
    
    if 'feedback' in raw:
        config.feedback = FeedbackConfig(**raw['feedback'])
    
    if 'regime' in raw:
        raw_regime = raw['regime']
        config.regime = RegimeConfig(
            open_time=raw_regime.get('time_regimes', {}).get('open', '10:30:00'),
            close_time=raw_regime.get('time_regimes', {}).get('close', '15:00:00')
        )
    
    if 'logging' in raw:
        config.logging = LoggingConfig(**raw['logging'])
    
    return config


def _get_default_config() -> AppConfig:
    """Return default configuration if file not found."""
    return AppConfig()


# Global config instance - lazy loaded
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance (singleton)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reset_config() -> None:
    """Reset the global config (useful for testing)."""
    global _config
    _config = None