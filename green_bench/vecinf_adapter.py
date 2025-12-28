from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LaunchedModel:
    name: str
    job_id: str
    base_url: str


def _import_vecinf_client():
    """
    vec_inf is a custom cluster library. Import paths may vary depending on your setup.
    """
    try:
        # Most likely
        from vec_inf import VecInfClient  # type: ignore

        return VecInfClient
    except Exception:
        try:
            from vec_inf.client import VecInfClient  # type: ignore

            return VecInfClient
        except Exception as e:
            raise ImportError(
                "Could not import VecInfClient. Tried: "
                "`from vec_inf import VecInfClient` and "
                "`from vec_inf.client import VecInfClient`. "
                "Ensure vec_inf is installed in this environment."
            ) from e


def _coerce_base_url(obj: Any) -> str:
    """
    vec_inf may return a raw string base_url or a dict-like payload.
    This function tries hard to normalize it into a string URL.
    """
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        for key in ("base_url", "server_address", "url", "endpoint"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip():
                return v
    # last resort
    return str(obj)


class VecInfLauncher:
    """
    Thin adapter around vec_inf's VecInfClient to make lifecycle handling explicit.
    """

    def __init__(self) -> None:
        VecInfClient = _import_vecinf_client()
        self._client = VecInfClient()

    def launch(self, model_name: str) -> str:
        """
        Returns a job_id (string).
        """
        return str(self._client.launch_model(model_name))

    def wait_until_ready(self, job_id: str) -> str:
        """
        Returns an OpenAI-compatible base_url.
        """
        res = self._client.wait_until_ready(job_id)
        return _coerce_base_url(res)

    def shutdown(self, job_id: str) -> None:
        self._client.shutdown_model(job_id)
