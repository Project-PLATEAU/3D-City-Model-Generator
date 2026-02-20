# BridgeUI Backend

A Python Flask backend server that simulates file processing for the BridgeUI frontend application.

## Features

- **File Upload Processing**: Handles GeoJSON, orthophoto (TIFF/images), and point cloud files
- **Async Processing Simulation**: Simulates realistic processing times for different file types
- **Progress Tracking**: Real-time progress updates for ongoing jobs
- **Output File Generation**: Creates mock 3D model files (OBJ/MTL) based on input type
- **Job Management**: Track, monitor, and clean up processing jobs

## API Endpoints

### Upload File
```http
POST /upload
Content-Type: multipart/form-data

Parameters:
- file: The uploaded file
- type: File type ('geojson', 'orthophoto', 'pointcloud')
- layerName: Name for the layer/output
```

### Check Job Status
```http
GET /jobs/{job_id}/status
```

### Download Result
```http
GET /jobs/{job_id}/download
```

### List All Jobs
```http
GET /jobs
```

### Delete Job
```http
DELETE /jobs/{job_id}
```

### Health Check
```http
GET /health
```

## File Processing Simulation

- **GeoJSON → LoD1**: 3 seconds processing → `.obj` file
- **Orthophoto → LoD2**: 8 seconds processing → `.obj` file  
- **Point Cloud → LoD3**: 15 seconds processing → `.obj` + `.mtl` files

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the server:
```bash
python server.py
```

The server will start on `http://localhost:5000`

## File Storage

- **Uploads**: Stored in `uploads/` directory
- **Outputs**: Generated in `outputs/{job_id}/` directories
- **Cleanup**: Jobs and files can be deleted via the API

## Integration with Frontend

The frontend (ImportModel.jsx) should be modified to:

1. Send uploaded files to `POST /upload`
2. Poll `GET /jobs/{job_id}/status` for progress
3. Download results from `GET /jobs/{job_id}/download` when complete

## Example Usage

```bash
# Upload a GeoJSON file
curl -X POST -F "file=@building_footprints.geojson" \
     -F "type=geojson" \
     -F "layerName=MyBuildings" \
     http://localhost:5000/upload

# Check status
curl http://localhost:5000/jobs/{job_id}/status

# Download result when complete
curl -O http://localhost:5000/jobs/{job_id}/download
```