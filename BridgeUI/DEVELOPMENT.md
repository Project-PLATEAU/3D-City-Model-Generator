# BridgeUI Development Guide

## Overview

BridgeUI is a web application with both frontend (React/Vite) and backend (Python Flask) components. The backend processes uploaded files and generates 3D models, while the frontend provides the user interface.

## Quick Start

### Option 1: Start Both Frontend and Backend Together (Recommended)

1. **Install Node.js dependencies:**
```bash
npm install
```

2. **Start both frontend and backend:**
```bash
npm run dev:full
```

This will automatically:
- Install Python dependencies for the backend
- Start the Flask backend server on `http://localhost:5000`
- Start the Vite development server on `http://localhost:5173`

### Option 2: Start Components Separately

#### Start Backend Only
```bash
npm run backend
```
or
```bash
python backend/start-backend.py
```

#### Start Frontend Only  
```bash
npm run dev
```

## Backend Integration

The frontend automatically detects if the backend is available and switches between:

- **Backend Processing** (when available): Files are uploaded to the Flask server for processing
- **Client-side Processing** (fallback): Files are processed in the browser (original behavior)

### Backend Features

- **File Upload**: Handles GeoJSON, orthophoto, and point cloud files
- **Processing Simulation**: Realistic processing times with progress updates
- **Output Generation**: Creates mock 3D model files (.obj/.mtl)
- **Job Management**: Track, monitor, and download processed results

### API Endpoints

The backend provides REST APIs at `http://localhost:5000`:

- `POST /upload` - Upload files for processing
- `GET /jobs/{id}/status` - Check processing progress
- `GET /jobs/{id}/download` - Download results
- `GET /jobs` - List all jobs
- `DELETE /jobs/{id}` - Clean up jobs
- `GET /health` - Backend health check

## File Processing Flow

1. **Upload**: User selects file in frontend
2. **Backend Check**: Frontend checks if backend is available
3. **Processing**: 
   - If backend available: Upload → Process → Download
   - If no backend: Client-side processing (original behavior)
4. **Results**: Processed files are integrated into the application

## Processing Times

- **GeoJSON → LoD1**: ~3 seconds
- **Orthophoto → LoD2**: ~8 seconds  
- **Point Cloud → LoD3**: ~15 seconds

## File Storage

- **Backend uploads**: `backend/uploads/`
- **Backend outputs**: `backend/outputs/{job_id}/`
- **Automatic cleanup**: Jobs can be deleted via API

## Development Tips

### Backend Development

- Backend auto-installs Python dependencies on first run
- Modify `backend/server.py` to change processing logic
- Add new file types in `ALLOWED_EXTENSIONS`
- Adjust processing times in `simulate_processing()`

### Frontend Development

- Backend integration in `src/services/backendApi.js`
- Upload UI in `src/components/ImportModel.jsx`
- Processing status shows real-time progress
- Fallback to client-side processing if backend unavailable

### Debugging

- Backend logs appear in console when started with `npm run dev:full`
- Frontend shows backend availability status in UI
- Check `http://localhost:5000/health` to verify backend status
- Network tab shows API requests to backend

## Environment Variables

No environment variables required - everything works with defaults:

- Frontend: `http://localhost:5173`
- Backend: `http://localhost:5000`
- Backend API detection is automatic

## Troubleshooting

### Backend Not Starting
- Check Python installation: `python --version`
- Install dependencies manually: `pip install -r backend/requirements.txt`
- Try starting backend separately: `python backend/server.py`

### Frontend Can't Connect to Backend
- Verify backend is running on port 5000
- Check browser console for CORS errors
- Backend health check: `curl http://localhost:5000/health`

### File Upload Issues
- Check file types are allowed in backend `ALLOWED_EXTENSIONS`
- Verify file size limits
- Check browser network tab for upload errors

## Production Deployment

For production deployment:

1. Build frontend: `npm run build`
2. Deploy backend with WSGI server (gunicorn, uwsgi)
3. Configure reverse proxy (nginx) for both services
4. Set appropriate CORS origins in backend
5. Use environment variables for configuration