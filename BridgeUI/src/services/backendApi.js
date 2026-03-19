/**
 * Backend API service for BridgeUI
 * Handles communication with the Python Flask backend
 */

// Use relative URL to go through Vite proxy (works with Cloudflare tunnel)
// In development, Vite proxies /api/* to localhost:5001
const BACKEND_URL = '';

class BackendApiService {
  constructor() {
    this.baseURL = BACKEND_URL;
  }

  /**
   * Upload a file to the backend for processing
   * @param {File} file - The file to upload
   * @param {string} type - File type ('geojson', 'orthophoto', 'pointcloud')
   * @param {string} layerName - Name for the layer
   * @returns {Promise<Object>} Upload response with job_id
   */
  async uploadFile(file, type, layerName) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('type', type);
    formData.append('layerName', layerName);

    try {
      console.log('📤 Uploading file to backend:', file.name, 'Type:', type);
      
      const response = await fetch(`${this.baseURL}/upload`, {
        method: 'POST',
        mode: 'cors',
        body: formData,
        // Don't set Content-Type header for FormData, let browser set it
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Upload failed');
      }

      const result = await response.json();
      console.log('✅ File uploaded successfully:', result);
      return result;
    } catch (error) {
      console.error('❌ Upload failed:', error);
      throw error;
    }
  }

  /**
   * Get the status of a processing job
   * @param {string} jobId - The job ID
   * @returns {Promise<Object>} Job status and progress
   */
  async getJobStatus(jobId) {
    try {
      const response = await fetch(`${this.baseURL}/jobs/${jobId}/status`, {
        method: 'GET',
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to get job status');
      }

      return await response.json();
    } catch (error) {
      console.error('❌ Failed to get job status:', error);
      throw error;
    }
  }

  /**
   * Download the processed result file
   * @param {string} jobId - The job ID
   * @returns {Promise<Blob>} The downloaded file as blob
   */
  async downloadResult(jobId) {
    try {
      const response = await fetch(`${this.baseURL}/jobs/${jobId}/download`);
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Download failed');
      }

      return await response.blob();
    } catch (error) {
      console.error('❌ Download failed:', error);
      throw error;
    }
  }

  /**
   * Get the download URL for a processed result
   * @param {string} jobId - The job ID
   * @returns {string} Download URL
   */
  getDownloadUrl(jobId) {
    return `${this.baseURL}/jobs/${jobId}/download`;
  }

  /**
   * List all jobs
   * @returns {Promise<Object>} List of all jobs
   */
  async listJobs() {
    try {
      const response = await fetch(`${this.baseURL}/jobs`, {
        method: 'GET',
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      if (!response.ok) {
        throw new Error('Failed to list jobs');
      }

      const data = await response.json();
      // Backend returns {jobs: [...], total: n}, we need the jobs array
      return Array.isArray(data) ? data : (data.jobs || []);
    } catch (error) {
      console.error('❌ Failed to list jobs:', error);
      throw error;
    }
  }

  /**
   * Delete a job and its files
   * @param {string} jobId - The job ID
   * @returns {Promise<Object>} Deletion confirmation
   */
  async deleteJob(jobId) {
    try {
      const response = await fetch(`${this.baseURL}/jobs/${jobId}`, {
        method: 'DELETE'
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to delete job');
      }

      return await response.json();
    } catch (error) {
      console.error('❌ Failed to delete job:', error);
      throw error;
    }
  }

  /**
   * Clear uploads and outputs directories on the backend
   * @returns {Promise<Object>} Cleanup confirmation
   */
  async cleanup() {
    try {
      const response = await fetch(`${this.baseURL}/cleanup`, {
        method: 'POST',
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Cleanup failed');
      }

      const result = await response.json();
      console.log('🧹 Cleanup completed:', result.message);
      return result;
    } catch (error) {
      console.error('❌ Cleanup failed:', error);
      throw error;
    }
  }

  /**
   * Generate LoD2 model from uploaded GeoJSON
   * @param {Object} config - LoD2 generation configuration
   * @param {string} config.height - Building height
   * @param {string} config.roofType - Roof type ('flat', 'gabled', etc.)
   * @param {string} [config.geojsonJobId] - Optional job ID of previously uploaded GeoJSON
   * @returns {Promise<Object>} Generation response with job_id
   */
  async generateLoD2Model(config) {
    console.log('🔧 Generating LoD2 model with config:', config);
    
    try {
      const response = await fetch(`${this.baseURL}/generate/lod2`, {
        method: 'POST',
        mode: 'cors',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      });

      console.log('📡 LoD2 generation response status:', response.status);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('❌ LoD2 generation failed with error:', errorData);
        throw new Error(errorData.error || 'LoD2 generation failed');
      }

      const result = await response.json();
      console.log('✅ LoD2 generation started successfully:', result);
      console.log('   ├─ Job ID:', result.job_id);
      console.log('   ├─ Expected OBJ path:', `/outputs/${result.job_id}/Untitled_lod2.obj`);
      console.log('   └─ Full URL:', `${this.baseURL}/outputs/${result.job_id}/Untitled_lod2.obj`);
      
      return result;
    } catch (error) {
      console.error('❌ LoD2 generation failed:', error);
      throw error;
    }
  }

  /**
   * Test if a specific OBJ file exists and is accessible
   * @param {string} jobId - The job ID
   * @param {string} fileName - Optional filename (defaults to 'Untitled_lod2.obj')
   * @returns {Promise<boolean>} True if file exists and is accessible
   */
  async testObjFileAccess(jobId, fileName = 'Untitled_lod2.obj') {
    const objUrl = `${this.baseURL}/outputs/${jobId}/${fileName}`;
    console.log('🔍 Testing OBJ file access:', objUrl);
    
    try {
      const response = await fetch(objUrl, { 
        method: 'HEAD',
        mode: 'cors',
      });
      const exists = response.ok;
      console.log(`   └─ File ${exists ? '✅ exists' : '❌ does not exist'} (status: ${response.status})`);
      return exists;
    } catch (error) {
      console.error('❌ Error testing OBJ file access:', error);
      return false;
    }
  }

  /**
   * Comprehensive test function for debugging backend integration
   */
  async debugBackendIntegration() {
    console.log('🧪 === BACKEND INTEGRATION DEBUG TEST ===');
    
    try {
      // 1. Test backend availability
      console.log('1️⃣ Testing backend availability...');
      const isAvailable = await this.isBackendAvailable();
      console.log(`   └─ Backend ${isAvailable ? '✅ available' : '❌ unavailable'}`);
      
      if (!isAvailable) {
        console.log('❌ Backend not available. Make sure Flask server is running on localhost:5001');
        return false;
      }
      
      // 2. List all jobs
      console.log('2️⃣ Listing all jobs...');
      const jobsData = await this.listJobs();
      console.log('   └─ Jobs data:', jobsData);
      
      const jobs = Array.isArray(jobsData) ? jobsData : (jobsData.jobs || []);
      console.log('   └─ Jobs array:', jobs);
      
      // 3. Find GeoJSON jobs  
      const geojsonJobs = jobs.filter(job => job.file_type === 'geojson');
      console.log('3️⃣ GeoJSON jobs:', geojsonJobs);
      
      // 4. Find orthophoto jobs (for LoD2 generation)
      const orthophotoJobs = jobs.filter(job => job.file_type === 'orthophoto');
      console.log('4️⃣ Orthophoto jobs (for LoD2):', orthophotoJobs);
      
      // 5. Find any existing LoD2 jobs or results
      const lod2Jobs = jobs.filter(job => job.file_type === 'lod2' || job.description?.includes('LoD2'));
      console.log('5️⃣ LoD2 jobs:', lod2Jobs);
      
      // 5. Look for LoD2 models in orthophoto job output directories
      console.log('5️⃣ Testing LoD2 models in orthophoto job outputs...');
      const lod2Results = [];
      
      for (const job of orthophotoJobs) {
        console.log('🔍 Checking job structure:', job);
        
        // Extract job ID - try different possible field names
        const jobId = job.id || job.job_id || job.ID || job.uuid;
        console.log('📋 Job ID found:', jobId);
        
        if (!jobId) {
          console.warn('⚠️ No job ID found for job:', job);
          continue;
        }

        // Construct expected filename using the job's layer_name
        const layerName = job.layer_name || 'Untitled';
        const expectedFileName = `${layerName}_lod2.obj`;

        const possibleFiles = [expectedFileName, 'Untitled_lod2.obj', 'lod2.obj'];
        
        for (const fileName of possibleFiles) {
          const accessible = await this.testObjFileAccess(jobId, fileName);
          if (accessible) {
            console.log(`   ├─ Job ${jobId} (${job.filename}): ✅ ${fileName} accessible`);
            lod2Results.push({
              jobId: jobId,
              filename: job.filename,
              lod2File: fileName,
              accessible: true
            });
            break; // Stop checking other files once we find one
          }
        }
      }
      
      console.log('📊 LoD2 Results Summary:', lod2Results);
      
      console.log('✅ Debug test completed successfully');
      return true;
      
    } catch (error) {
      console.error('❌ Debug test failed:', error);
      return false;
    }
  }

  /**
   * Upload LoD3 files (pointcloud folder + streetview folder) to the backend
   * @param {Object} files - Object containing pointcloud and streetview file arrays
   * @param {Object} files.pointcloud - Pointcloud folder info with files array
   * @param {Object} files.streetview - Streetview folder info with files array
   * @param {string} layerName - Name for the layer
   * @returns {Promise<Object>} Upload response with job_id
   */
  async uploadLod3Files(files, layerName) {
    const formData = new FormData();

    // Add pointcloud files
    const pcFiles = files.pointcloud?.files || [];
    formData.append('pointcloud_count', pcFiles.length);
    formData.append('pointcloud_folder', files.pointcloud?.folderName || 'pointcloud');
    pcFiles.forEach((file, index) => {
      formData.append(`pointcloud_${index}`, file);
    });

    // Add streetview files
    const svFiles = files.streetview?.files || [];
    formData.append('streetview_count', svFiles.length);
    formData.append('streetview_folder', files.streetview?.folderName || 'streetview');
    svFiles.forEach((file, index) => {
      formData.append(`streetview_${index}`, file);
    });

    // Add layer name
    formData.append('layerName', layerName || 'Combined_LoD3');

    try {
      console.log('📤 Uploading LoD3 files to backend:');
      console.log(`   ├─ Pointcloud: ${pcFiles.length} files`);
      console.log(`   └─ Streetview: ${svFiles.length} files`);

      const response = await fetch(`${this.baseURL}/upload-lod3`, {
        method: 'POST',
        mode: 'cors',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'LoD3 upload failed');
      }

      const result = await response.json();
      console.log('✅ LoD3 files uploaded successfully:', result);
      return result;
    } catch (error) {
      console.error('❌ LoD3 upload failed:', error);
      throw error;
    }
  }

  /**
   * Check if the backend is available
   * @returns {Promise<boolean>} True if backend is healthy
   */
  async isBackendAvailable() {
    try {
      console.log('🔍 Testing backend connection to:', `${this.baseURL}/health`);
      
      const response = await fetch(`${this.baseURL}/health`, {
        method: 'GET',
        mode: 'cors', // Explicitly request CORS
        headers: {
          'Content-Type': 'application/json',
        },
      });
      
      const isHealthy = response.ok;
      console.log(`🏥 Backend health check: ${isHealthy ? '✅ healthy' : '❌ unhealthy'} (status: ${response.status})`);
      
      if (!isHealthy && response.status === 403) {
        console.error('❌ CORS Error: Backend is not configured to accept requests from this origin');
        console.error('   Fix: Add CORS headers to your Flask backend');
        console.error('   Add this to your Flask app:');
        console.error('   from flask_cors import CORS');
        console.error('   CORS(app, origins=["http://localhost:5173"])');
      }
      
      return isHealthy;
    } catch (error) {
      console.warn('❌ Backend connection failed:', error.message);
      
      if (error.message.includes('CORS')) {
        console.error('🚫 CORS Issue Detected:');
        console.error('   Your Flask backend needs CORS configuration');
        console.error('   Install: pip install flask-cors');
        console.error('   Add to your Flask app:');
        console.error('   from flask_cors import CORS');
        console.error('   CORS(app, origins=["http://localhost:5173"])');
      } else if (error.message.includes('Failed to fetch') || error.message.includes('Load failed')) {
        console.error('🔌 Connection Issue:');
        console.error('   Backend might not be running on localhost:5001');
        console.error('   Or there might be a network/firewall issue');
      }
      
      return false;
    }
  }

  /**
   * Poll job status until completion
   * @param {string} jobId - The job ID
   * @param {Function} onProgress - Progress callback
   * @param {number} intervalMs - Polling interval in milliseconds (default: 3 seconds)
   * @returns {Promise<Object>} Final job status when completed
   */
  async pollJobStatus(jobId, onProgress = null, intervalMs = 3000) {
    return new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const status = await this.getJobStatus(jobId);
          console.log(`🔄 Poll status for job ${jobId}:`, status);

          if (onProgress) {
            console.log(`🔄 Calling onProgress callback with status`);
            onProgress(status);
          }

          if (status.status === 'completed') {
            resolve(status);
            return;
          }

          if (status.status === 'failed') {
            reject(new Error('Job processing failed'));
            return;
          }

          // Continue polling
          setTimeout(poll, intervalMs);
        } catch (error) {
          reject(error);
        }
      };

      poll();
    });
  }
}

// Create a singleton instance
const backendApi = new BackendApiService();

// Add debug function to window for easy testing
if (typeof window !== 'undefined') {
  window.debugBackend = () => backendApi.debugBackendIntegration();
  window.testObjFile = (jobId, fileName = 'Untitled_lod2.obj') => backendApi.testObjFileAccess(jobId, fileName);
  window.listBackendJobs = () => backendApi.listJobs();
  
  // Helper function to test with known working job ID
  window.testWithKnownJobId = () => {
    // Test with the job ID we know exists from your outputs folder
    const knownJobId = 'ce2dc3a6-59ed-4ca0-8a5b-63297eb4d8f2';
    console.log('🧪 Testing with known job ID:', knownJobId);
    return backendApi.testObjFileAccess(knownJobId, '1_lod2.obj');
  };
  
  // Function to manually add existing LoD2 models to the app
  window.addExistingLoD2Models = async () => {
    try {
      console.log('🔍 Manually adding existing LoD2 models...');
      
      const jobs = await backendApi.listJobs();
      const orthophotoJobs = jobs.filter(job => job.file_type === 'orthophoto' && job.completed_at);
      
      console.log('📋 Found orthophoto jobs:', orthophotoJobs.length);
      
      const foundModels = [];
      
      for (const job of orthophotoJobs) {
        // Get job ID - check multiple possible field names
        const jobId = job.id || job.job_id || job.ID || Object.keys(job).find(key => 
          key.toLowerCase().includes('id') && job[key] && typeof job[key] === 'string'
        );
        
        if (!jobId) {
          console.log('⚠️ No job ID found for job:', job);
          continue;
        }
        
        console.log(`🔍 Checking job ${jobId} (${job.filename})`);

        // Construct expected filename using the job's layer_name
        const layerName = job.layer_name || 'Untitled';
        const expectedFileName = `${layerName}_lod2.obj`;

        // Check for LoD2 files
        const possibleFiles = [expectedFileName, 'Untitled_lod2.obj', 'lod2.obj'];
        
        for (const fileName of possibleFiles) {
          const exists = await backendApi.testObjFileAccess(jobId, fileName);
          if (exists) {
            console.log(`✅ Found ${fileName} for job ${jobId}`);
            foundModels.push({
              jobId,
              fileName,
              sourceFilename: job.filename,
              createdAt: job.completed_at,
              objPath: `/outputs/${jobId}/${fileName}`,
              fullUrl: `/outputs/${jobId}/${fileName}`
            });
            break;
          }
        }
      }
      
      console.log('🎯 Found LoD2 models:', foundModels);
      
      // Try to trigger app-level layer addition
      if (foundModels.length > 0) {
        console.log('📢 Call window.appInstance.checkAndAddExistingLoD2Models() or manually add layers through the import dialog.');
        console.log('💡 Or refresh the page and the models should be auto-detected.');
      }
      
      return foundModels;
      
    } catch (error) {
      console.error('Failed to add existing LoD2 models:', error);
      return [];
    }
  };
}

export default backendApi;