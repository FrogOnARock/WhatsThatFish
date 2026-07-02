from fastapi import APIRouter, Depends

from ..dependencies import get_current_user, get_user_service
from ..schemas import UserProfile, UserSettingsUpdate
from ..services.service import UserService
from whatsthatfish.database.models import User

router = APIRouter()


@router.get("/auth/me", response_model=UserProfile)
def me(
    user: User = Depends(get_current_user),
    svc: UserService = Depends(get_user_service),
) -> UserProfile:
    """Return the signed-in user's profile. The SPA calls this after Google
    sign-in to confirm the session and hydrate the avatar/name; the upsert in
    get_current_user makes first call double as account creation."""
    return svc.profile(user)


@router.patch("/auth/me", response_model=UserProfile)
def update_me(
    data: UserSettingsUpdate,
    user: User = Depends(get_current_user),
    svc: UserService = Depends(get_user_service),
) -> UserProfile:
    """Edit the app-owned profile fields (preferred_name / unit_system). The
    Google-sourced name/email/avatar are read-only here, since the login sync
    overwrites them every time."""
    return svc.update_settings(user, data)
