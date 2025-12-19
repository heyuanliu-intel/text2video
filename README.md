# Text2Video Service

OPEA Text-to-Video microservice for generating videos from text prompts and audio inputs.

## Overview

This service provides a standalone deployment of the OPEA text2video component. It exposes REST APIs for video generation using state-of-the-art diffusion models optimized for Intel Habana Gaudi accelerators.

## Features

- Text-to-video generation with audio conditioning
- Job queue management for concurrent requests
- HPU/Gaudi optimization support
- RESTful API with OpenAPI documentation
- Docker deployment support

## Installation

### From PyPI

```bash
pip install opea-comps
pip install text2video
```

### From Source

```bash
git clone <repository-url>
cd text2video
pip install -e .
```

## Usage

### Starting the Service

```bash
text2video-service --model_name_or_path InfinteTalk --video_dir /tmp/videos
```

### API Endpoints

- **POST /v1/videos** - Create a new video generation job
- **GET /v1/videos/{video_id}** - Get job status
- **GET /v1/videos/{video_id}/content** - Download completed video
- **DELETE /v1/videos/{video_id}** - Delete a video job

### Example Request

```python
import requests

url = "http://localhost:9396/v1/videos"
files = {
    "prompt": (None, "A beautiful sunset over the ocean"),
    "input_reference": open("reference.jpg", "rb"),
    "audio[]": open("audio.wav", "rb"),
}
response = requests.post(url, files=files)
print(response.json())
```

## Docker Deployment

```bash
docker-compose up
```

## Development

Install development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest tests/
```

## License

Apache 2.0

## Links

- [OPEA Project](https://github.com/opea-project/GenAIComps)
- [Documentation](https://opea-project.github.io/)
