# models.py
from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Boolean,
    Float,
    ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

class Critic(Base):
    __tablename__ = 'critics'
    id = Column(Integer, primary_key=True)
    critic_name = Column(String, nullable=False)
    
    ratings = relationship("Rating", back_populates="critic")
    games_nominated = relationship("Game", back_populates="nominator")

class Game(Base):
    __tablename__ = 'games'
    id = Column(Integer, primary_key=True)
    game_name = Column(String, nullable=False)
    upcoming = Column(Boolean, default=False)
    nominated_by = Column(Integer, ForeignKey('critics.id'))
    
    nominator = relationship("Critic", back_populates="games_nominated")
    ratings = relationship("Rating", back_populates="game")

class Rating(Base):
    __tablename__ = 'ratings'
    id = Column(Integer, primary_key=True)
    critic_id = Column(Integer, ForeignKey('critics.id'), nullable=False)
    game_id = Column(Integer, ForeignKey('games.id'), nullable=False)
    score = Column(Float)

    critic = relationship("Critic", back_populates="ratings")
    game = relationship("Game", back_populates="ratings")