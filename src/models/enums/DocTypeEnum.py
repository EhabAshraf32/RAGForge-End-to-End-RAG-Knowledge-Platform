from enum import Enum

class DocTypeEnum(Enum):

    LECTURE = "lecture"
    LAB = "lab"
    OTHER = "other"

    @classmethod
    def normalize(cls, value: str):
        """Return the canonical doc type value ('lecture' / 'lab' / 'other'), or None if invalid."""
        if value is None:
            return None
        value = str(value).strip().lower()
        for item in cls:
            if item.value == value:
                return item.value
        return None
