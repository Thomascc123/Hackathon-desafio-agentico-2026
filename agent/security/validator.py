from pydantic import BaseModel, Field, field_validator


class UserQuery(BaseModel):
    text: str = Field(
        ...,
        max_length=500,
        min_length=1,
        description="User query text"
    )

    @field_validator("text")
    @classmethod
    def sanitize(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Query cannot be empty")
        return stripped
