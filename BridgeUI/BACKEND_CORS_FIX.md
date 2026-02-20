# Backend CORS Configuration Fix

## Problem
The frontend running on `http://localhost:5173` cannot connect to the Flask backend on `http://localhost:5000` due to CORS (Cross-Origin Resource Sharing) restrictions.

Error message:
```
Origin http://localhost:5173 is not allowed by Access-Control-Allow-Origin. Status code: 403
```

## Solution

### 1. Install flask-cors
```bash
pip install flask-cors
```

### 2. Add CORS configuration to your Flask app

**Option A: Allow specific origin (Recommended for development)**
```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# Allow requests from the frontend
CORS(app, origins=["http://localhost:5173"])

# OR with more specific configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173"],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

**Option B: Allow all origins (Not recommended for production)**
```python
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # This allows all origins
```

**Option C: Manual CORS headers (if you prefer not to use flask-cors)**
```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/health', methods=['OPTIONS'])
def handle_options():
    response = jsonify({'status': 'ok'})
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response
```

### 3. Required endpoints for BridgeUI integration

Make sure your Flask backend has these endpoints:

```python
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/upload', methods=['POST'])
def upload_file():
    # Handle file upload
    pass

@app.route('/generate/lod2', methods=['POST'])
def generate_lod2():
    # Handle LoD2 generation
    pass

@app.route('/jobs', methods=['GET'])
def list_jobs():
    # List all jobs
    pass

@app.route('/jobs/<job_id>/status', methods=['GET'])
def get_job_status(job_id):
    # Get job status
    pass

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    # Serve generated OBJ files
    return send_from_directory('outputs', filename)
```

### 4. Testing the fix

After adding CORS configuration:

1. Restart your Flask backend
2. Refresh the BridgeUI frontend
3. Open browser console and run:
   ```javascript
   await debugBackend()
   ```
4. You should see:
   ```
   🏥 Backend health check: ✅ healthy (status: 200)
   ```

### 5. File serving configuration

Make sure your Flask app can serve OBJ files from the outputs directory:

```python
from flask import send_from_directory
import os

# Create outputs directory if it doesn't exist
if not os.path.exists('outputs'):
    os.makedirs('outputs')

@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    return send_from_directory('outputs', filename)
```

## Verification

Once CORS is properly configured, the frontend should be able to:
- Check backend health (`/health`)
- Upload files (`/upload`) 
- Generate LoD2 models (`/generate/lod2`)
- Poll job status (`/jobs/<id>/status`)
- Access generated OBJ files (`/outputs/<job_id>/Untitled_lod2.obj`)

## Security Note

For production deployment:
- Never use `CORS(app)` without restrictions
- Always specify allowed origins explicitly
- Consider using environment variables for configuration:

```python
import os
from flask_cors import CORS

allowed_origins = os.getenv('ALLOWED_ORIGINS', 'http://localhost:5173').split(',')
CORS(app, origins=allowed_origins)
```