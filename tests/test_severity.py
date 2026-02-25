import os
import pytest

try:
    from src.services.severity_service import predict_severity
    HAS_ULTRA = True
except Exception:
    HAS_ULTRA = False


@pytest.mark.skipif(not HAS_ULTRA, reason="Ultralytics not installed or service unavailable")
def test_severity_service_runs_on_image(tmp_path):
    # Create a tiny dummy image
    p = tmp_path / "dummy.jpg"
    from PIL import Image
    import numpy as np
    img = Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8"))
    img.save(p.as_posix())

    # Skip if weights not present
    weights_default = os.path.join("models", "severity_model.pt")
    if not os.path.exists(weights_default):
        pytest.skip("Severity weights not found at models/severity_model.pt")

    res = predict_severity(p.as_posix())
    assert isinstance(res, dict)
*** End Patch