from fastapi import Request, Depends, Header
from sqlalchemy.orm import Session

from ..database.config import get_session_factory
from ..database.models import User
from whatsthatfish.serving.services.service import (
    PredictionService,
    UserService,
    ObservationService,
)
from .auth import verify_google_token
from .error import AuthenticationException
from .utils import ContributionConstructor


_session_factory = None


def _get_factory():
    # Built lazily on first request rather than at import, so importing this
    # module (e.g. during pytest collection) never requires DATABASE_URL.
    global _session_factory
    if _session_factory is None:
        _session_factory = get_session_factory()
    return _session_factory


def get_session():
    with _get_factory()() as session:
        yield session


def get_current_user(
    authorization: str | None = Header(default=None),
    session: Session = Depends(get_session),
) -> User:
    """Auth gate: verify the Bearer Google ID token and resolve the local User
    (creating it on first sign-in). Protect a route by adding this as a Depends;
    the User is then injected and ownership scoping reads `user.id`."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationException("Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    claims = verify_google_token(token)
    return UserService(session).get_or_create(claims)


def get_user_service(
    session: Session = Depends(get_session),
) -> UserService:
    return UserService(session)


def get_prediction_service(
    request: Request, session: Session = Depends(get_session)
) -> PredictionService:
    return PredictionService(
        session=session,
        bbox_inferrer=request.app.state.bbox_inferrer,
        class_inferrer=request.app.state.class_inferrer,
    )


def get_observation_service(
    session: Session = Depends(get_session),
) -> ObservationService:
    # Contribution storage is built per-request (cheap locally; mirrors how the
    # library router constructs image storage).
    return ObservationService(
        session=session, storage=ContributionConstructor().constructor()
    )
