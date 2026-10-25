# Same base image as the Codabench evaluation container.
FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

# The submission zip already contains everything needed for inference
# (run.py + metadata.yaml + assets/par_model.pt). Pillow ships with the base
# image; nothing else is required at inference time.
WORKDIR /workspace

COPY submission_zip/task1_clip_vitb16_cal.zip /workspace/task1.zip
COPY run.py /workspace/reference_run.py
COPY validate_zip.py /workspace/validate_zip.py

# Optional: local val-split verification (organizers provide the data).
# Mount the UPAR data dir at /data, i.e.
#   docker run --gpus all -v /path/to/upar-data:/data upar2027-track1
# The runner below evaluates the EXACT submission zip through the same
# code path as the Codabench ingestion program.
RUN printf '#!/bin/bash\n\
set -e\n\
mkdir -p /workspace/ingest\n\
cd /workspace/ingest\n\
unzip -o /workspace/task1.zip\n\
python - <<EOF\n\
import importlib.util, json\n\
spec = importlib.util.spec_from_file_location("run", "/workspace/ingest/run.py")\n\
run = importlib.util.module_from_spec(spec)\n\
spec.loader.exec_module(run)\n\
run.load_model()\n\
import numpy as np, sys\n\
sys.path.insert(0, "/workspace")\n\
# small self-test on one image if data is mounted\n\
import os\n\
if os.path.isdir("/data"):\n\
    from pathlib import Path\n\
    DATA = Path("/data")\n\
    gt_csv = DATA / "annotations/task1/val/gt.csv"\n\
    if gt_csv.exists():\n\
        print("val split found; running validate_zip.py")\n\
        os._exit(0)\n\
    else:\n\
        print("no val split under /data; smoke-test only")\n\
sample = {"image_path": None, "attribute_names": [str(i) for i in range(40)]}\n\
print("load_model OK; model loaded:", run._MODEL is not None)\n\
EOF\n\
if [ -d /data ] && [ -f /data/annotations/task1/val/gt.csv ]; then\n\
  cd /workspace && python validate_zip.py task1 /workspace/task1.zip\n\
fi\n' > /workspace/run.sh && chmod +x /workspace/run.sh

CMD ["/workspace/run.sh"]
