# database_models.py
from sqlalchemy import (
    Column,
    Boolean,
    Float,
    ForeignKey,
    Text,
    BigInteger
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Critic(Base):
    """Represents a critic who rates games."""
    __tablename__ = 'critics'
    id = Column(BigInteger, primary_key=True)
    critic_name = Column(Text, nullable=False, unique=True)
    
    # Relationships
    ratings = relationship("Rating", back_populates="critic")
    games_nominated = relationship("Game", back_populates="nominator")
    predictions = relationship("CriticPrediction", back_populates="critic")
    feature_importances = relationship("CriticFeatureImportance", back_populates="critic")

class Game(Base):
    """Represents a video game to be rated."""
    __tablename__ = 'games'
    id = Column(BigInteger, primary_key=True)
    game_name = Column(Text)
    upcoming = Column(Boolean, default=False)
    nominated_by = Column(BigInteger, ForeignKey('critics.id'))
    
    # Relationships
    nominator = relationship("Critic", back_populates="games_nominated")
    ratings = relationship("Rating", back_populates="game")
    predictions = relationship("CriticPrediction", back_populates="game")

class Rating(Base):
    """Represents a score a critic has given to a game."""
    __tablename__ = 'ratings'
    id = Column(BigInteger, primary_key=True)
    critic_id = Column(BigInteger, ForeignKey('critics.id'), nullable=False)
    game_id = Column(BigInteger, ForeignKey('games.id'), nullable=False)
    score = Column(Float)

    # Relationships
    critic = relationship("Critic", back_populates="ratings")
    game = relationship("Game", back_populates="ratings")

class CriticPrediction(Base):
    """Stores the model's predicted score and skip probability for a critic-game pair."""
    __tablename__ = 'critic_predictions'
    
    # Composite primary key linking a prediction to a specific game and critic
    id = Column(BigInteger, ForeignKey('games.id'), primary_key=True)
    critic_id = Column(BigInteger, ForeignKey('critics.id'), primary_key=True)
    
    name = Column(Text) # Game name, likely for easier lookups
    predicted_score = Column(Float)
    predicted_skip_probability = Column(Float)
    
    # Relationships
    critic = relationship("Critic", back_populates="predictions")
    game = relationship("Game", back_populates="predictions")

class CriticFeatureImportance(Base):
    """Stores the feature importance values for each critic's prediction models."""
    __tablename__ = 'critic_feature_importances'
    
    # Assumed composite primary key, as one is required for the ORM
    critic_id = Column(BigInteger, ForeignKey('critics.id'), primary_key=True)
    model_type = Column(Text, primary_key=True)
    feature = Column(Text, primary_key=True)
    
    importance = Column(Float)
    
    # Relationship
    critic = relationship("Critic", back_populates="feature_importances")

class CriticPredictionExplanation(Base):
    """Stores the pre-computed SHAP values for each prediction."""
    __tablename__ = 'critic_prediction_explanations'
    critic_id = Column(BigInteger, ForeignKey('critics.id'), primary_key=True)
    game_id = Column(BigInteger, ForeignKey('games.id'), primary_key=True)
    model_type = Column(Text, primary_key=True)
    base_value = Column(Float)
    shap_values = Column(Text) # Stored as a JSON string