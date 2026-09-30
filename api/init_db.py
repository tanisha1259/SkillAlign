from api.db import Base, engine
from api.models import Analysis, MatchResult


Base.metadata.create_all(bind=engine)

print("SkillAlign database tables created.")