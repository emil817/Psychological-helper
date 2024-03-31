import sqlalchemy as db
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import inspect

# Database
# DB.test: 0 - Agression, 1 - Anxiety, 2 - Depression

class Base(DeclarativeBase):
    pass

class User(Base):
    __tablename__ = 'users'
    id: Mapped[int] = mapped_column(primary_key=True)
    tg_id: Mapped[str]
    count_depressed_messages: Mapped[int]
    messages_history: Mapped[str | None]
    test: Mapped[int]

class SQL_DB():
    def __init__(self) -> None:
        self.engine = db.create_engine('sqlite:///database.db')
        self.sessionmaker = sessionmaker(self.engine)
        if not inspect(self.engine).has_table("users"):
            self.create_tables()

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def insert_user(self, user_data: User) -> None:
        with self.sessionmaker() as session:
            session.add(user_data)
            session.commit()

    def get_user(self, tg_id) -> User:
        with self.sessionmaker() as session:
            return session.query(User).filter(User.tg_id == tg_id).first()
    
    def update_user(self, user_data: User) -> None:
        with self.sessionmaker() as session:
            session.query(User).filter(User.tg_id == user_data.tg_id).update({
                'count_depressed_messages': user_data.count_depressed_messages,
                'messages_history': user_data.messages_history,
                'test': user_data.test
            })
            session.commit()
