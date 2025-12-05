# database_models.py
from sqlalchemy import (
    Column,
    Boolean,
    Float,
    ForeignKey,
    Text,
    BigInteger
)
from sqlalchemy.types import JSON  # Needed for user_tags
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
    
    # New: Link to details (One-to-One)
    details = relationship("GameDetails", uselist=False, back_populates="game")

class GameDetails(Base):
    """
    Stores metadata (Steam tags, price, release date) scraped from Steam.
    Required for the new content-based prediction system.
    """
    __tablename__ = 'games_details'
    
    # In your Supabase schema, 'id' is the FK to games.id and acts as PK
    id = Column(BigInteger, ForeignKey('games.id'), primary_key=True)
    
    appid = Column(BigInteger)
    name = Column(Text)
    user_tags = Column(JSON) # Stores the list of tags e.g. ["RPG", "Indie"]
    price_usd = Column(Float)
    release_date = Column(Text)
    developer_genres = Column(Text)
    developers = Column(Text)
    publishers = Column(Text)
    
    # Relationship
    game = relationship("Game", back_populates="details")

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
    """Stores the model's predicted score and skip probability."""
    __tablename__ = 'critic_predictions'
    
    # Composite primary key
    id = Column(BigInteger, ForeignKey('games.id'), primary_key=True)
    critic_id = Column(BigInteger, ForeignKey('critics.id'), primary_key=True)
    
    name = Column(Text)
    predicted_score = Column(Float)
    predicted_skip_probability = Column(Float)
    
    # Relationships
    critic = relationship("Critic", back_populates="predictions")
    game = relationship("Game", back_populates="predictions")

class CriticFeatureImportance(Base):
    """
    Stores the 'Affinity Profile' for each critic.
    'model_type' is now primarily 'relative_affinity'.
    'importance' stores the affinity score (e.g., +1.5 or -0.5).
    """
    __tablename__ = 'critic_feature_importances'
    
    critic_id = Column(BigInteger, ForeignKey('critics.id'), primary_key=True)
    model_type = Column(Text, primary_key=True) # e.g., 'relative_affinity'
    feature = Column(Text, primary_key=True)    # e.g., 'tag__RPG' or 'bin__Price_Low'
    
    importance = Column(Float)
    
    # Relationship
    critic = relationship("Critic", back_populates="feature_importances")