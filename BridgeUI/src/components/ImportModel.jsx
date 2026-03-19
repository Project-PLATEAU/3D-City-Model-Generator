import React, { useState, useRef, useEffect } from 'react';
import '../styles/Components.css';
import { Icon } from '../utils/iconLoader';
import backendApi from '../services/backendApi';
import { fromBlob } from 'geotiff';

const ImportModel = ({ isOpen, onClose, onImport, projectStage = 'initial', availableLoDs = new Set() }) => {
  const [layerName, setLayerName] = useState('');
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const [fileStatus, setFileStatus] = useState('idle'); // 'idle' | 'loading' | 'success' | 'error' | 'processing' | 'backend-processing'
  const [fileError, setFileError] = useState('');
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedType, setSelectedType] = useState(null); // null | 'geojson' | 'orthophoto' | 'pointcloud'
  const [showTypeSelection, setShowTypeSelection] = useState(true);
  const [showLoDSelection, setShowLoDSelection] = useState(false); // New state for LoD selection
  const [selectedLoD, setSelectedLoD] = useState(null); // Track selected LoD
  const [lodConfig, setLoDConfig] = useState({ height: '', roofType: 'flat' }); // LoD configuration
  const [useBackend, setUseBackend] = useState(true); // Toggle between backend and client-side processing
  const [backendAvailable, setBackendAvailable] = useState(false);
  const [currentJobId, setCurrentJobId] = useState(null);
  const [processingStatus, setProcessingStatus] = useState('');
  const [geojsonData, setGeojsonData] = useState(null); // Store parsed GeoJSON data for preview
  const fileInputRef = useRef(null);

  // ETA tracking for LoD2 building generation (40-90% progress range)
  const buildingPhaseStartTime = useRef(null);
  const buildingPhaseStartProgress = useRef(null);
  const estimatedTotalBuildings = useRef(null);

  // ETA tracking for LoD3 building generation (30-80% progress range)
  const lod3PhaseStartTime = useRef(null);
  const lod3PhaseStartProgress = useRef(null);

  // Calculate ETA for LoD3 building generation phase (30-80%)
  const calculateLod3ETA = (currentProgress) => {
    const LOD3_PHASE_START = 30;
    const LOD3_PHASE_END = 80;

    // Only calculate ETA during building generation phase (30-80%)
    if (currentProgress < LOD3_PHASE_START || currentProgress >= LOD3_PHASE_END) {
      lod3PhaseStartTime.current = null;
      lod3PhaseStartProgress.current = null;
      return null;
    }

    const now = Date.now();

    // Initialize tracking when entering building phase
    if (lod3PhaseStartTime.current === null) {
      lod3PhaseStartTime.current = now;
      lod3PhaseStartProgress.current = currentProgress;
      return null;
    }

    const elapsedMs = now - lod3PhaseStartTime.current;
    const progressInPhase = currentProgress - lod3PhaseStartProgress.current;

    // Need some progress to estimate
    if (progressInPhase < 1 || elapsedMs < 500) {
      return null;
    }

    // Estimate remaining time based on progress rate
    const msPerPercent = elapsedMs / progressInPhase;
    const remainingProgress = LOD3_PHASE_END - currentProgress;
    const remainingSeconds = Math.ceil((msPerPercent * remainingProgress) / 1000);

    // Format ETA
    if (remainingSeconds < 60) {
      return `${remainingSeconds}秒`;
    } else if (remainingSeconds < 3600) {
      const minutes = Math.floor(remainingSeconds / 60);
      const seconds = remainingSeconds % 60;
      return `${minutes}分${seconds}秒`;
    } else {
      const hours = Math.floor(remainingSeconds / 3600);
      const minutes = Math.floor((remainingSeconds % 3600) / 60);
      return `${hours}時間${minutes}分`;
    }
  };

  // Calculate ETA for LoD2 building generation phase (40-90%)
  // Each building takes approximately 0.8 seconds
  const calculateBuildingETA = (currentProgress) => {
    const SECONDS_PER_BUILDING = 0.8;
    const BUILDING_PHASE_START = 40;
    const BUILDING_PHASE_END = 90;
    const BUILDING_PHASE_RANGE = BUILDING_PHASE_END - BUILDING_PHASE_START; // 50%

    // Only calculate ETA during building generation phase (40-90%)
    if (currentProgress < BUILDING_PHASE_START || currentProgress >= BUILDING_PHASE_END) {
      buildingPhaseStartTime.current = null;
      buildingPhaseStartProgress.current = null;
      return null;
    }

    const now = Date.now();

    // Initialize tracking when entering building phase
    if (buildingPhaseStartTime.current === null) {
      buildingPhaseStartTime.current = now;
      buildingPhaseStartProgress.current = currentProgress;
      return null;
    }

    const elapsedMs = now - buildingPhaseStartTime.current;
    const progressInPhase = currentProgress - buildingPhaseStartProgress.current;

    // Need some progress to estimate
    if (progressInPhase < 1 || elapsedMs < 500) {
      return null;
    }

    // Estimate remaining time based on progress rate
    const msPerPercent = elapsedMs / progressInPhase;
    const remainingProgress = BUILDING_PHASE_END - currentProgress;
    const remainingSeconds = Math.ceil((msPerPercent * remainingProgress) / 1000);

    // Format ETA
    if (remainingSeconds < 60) {
      return `${remainingSeconds}秒`;
    } else if (remainingSeconds < 3600) {
      const minutes = Math.floor(remainingSeconds / 60);
      const seconds = remainingSeconds % 60;
      return `${minutes}分${seconds}秒`;
    } else {
      const hours = Math.floor(remainingSeconds / 3600);
      const minutes = Math.floor((remainingSeconds % 3600) / 60);
      return `${hours}時間${minutes}分`;
    }
  };

  useEffect(() => {
    if (previewUrl) {
      return () => {
        URL.revokeObjectURL(previewUrl);
      };
    }
  }, [previewUrl]);

  // Check backend availability on component mount
  useEffect(() => {
    const checkBackend = async () => {
      console.log('🔍 Checking backend availability...');
      const available = await backendApi.isBackendAvailable();
      console.log('🏥 Backend availability check result:', available);
      setBackendAvailable(available);
      if (!available) {
        console.warn('⚠️ Backend not available, falling back to client-side processing');
        setUseBackend(false);
      } else {
        console.log('✅ Backend is available and ready');
      }
    };
    
    if (isOpen) {
      console.log('📂 ImportModel opened, checking backend...');
      checkBackend();
    }
  }, [isOpen]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setSelectedType(null);
      setShowTypeSelection(true);
      setShowLoDSelection(false);
      setSelectedLoD(null);
      setLoDConfig({ height: '', roofType: 'flat' });
      setLayerName('');
      setFile(null);
      setFileStatus('idle');
      setFileError('');
      setPreviewUrl(null);
      setGeojsonData(null);
      setProgress(0);
      setProcessingStatus('');
      // Reset ETA tracking for building phase
      buildingPhaseStartTime.current = null;
      buildingPhaseStartProgress.current = null;
      estimatedTotalBuildings.current = null;
      // Reset ETA tracking for LoD3 building phase
      lod3PhaseStartTime.current = null;
      lod3PhaseStartProgress.current = null;
    }
  }, [isOpen]);

  if (!isOpen) return null;

  // Helper function to extract filename without extension for auto-fill
  const getFilenameWithoutExtension = (filename) => {
    const lastDotIndex = filename.lastIndexOf('.');
    if (lastDotIndex === -1) return filename; // No extension
    return filename.substring(0, lastDotIndex);
  };

  // Function to create GeoJSON preview as SVG
  const createGeojsonPreview = (geojsonData) => {
    try {
      if (!geojsonData || !geojsonData.features || geojsonData.features.length === 0) {
        return null;
      }

      // Calculate bounds
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      
      const processCoordinates = (coords) => {
        if (typeof coords[0] === 'number') {
          const [x, y] = coords;
          minX = Math.min(minX, x);
          maxX = Math.max(maxX, x);
          minY = Math.min(minY, y);
          maxY = Math.max(maxY, y);
        } else {
          coords.forEach(processCoordinates);
        }
      };

      geojsonData.features.forEach(feature => {
        if (feature.geometry && feature.geometry.coordinates) {
          processCoordinates(feature.geometry.coordinates);
        }
      });

      // Check if bounds are valid
      if (!isFinite(minX) || !isFinite(minY) || !isFinite(maxX) || !isFinite(maxY)) {
        return null;
      }

      // Add padding
      const padding = 0.1;
      const width = maxX - minX;
      const height = maxY - minY;
      const paddingX = width * padding;
      const paddingY = height * padding;

      minX -= paddingX;
      maxX += paddingX;
      minY -= paddingY;
      maxY += paddingY;

      // SVG dimensions
      const svgWidth = 300;
      const svgHeight = 200;

      // Scale to fit SVG
      const scaleX = svgWidth / (maxX - minX);
      const scaleY = svgHeight / (maxY - minY);
      const scale = Math.min(scaleX, scaleY);

      // Center the geometry
      const offsetX = (svgWidth - (maxX - minX) * scale) / 2;
      const offsetY = (svgHeight - (maxY - minY) * scale) / 2;

      const transformCoordinates = (coords) => {
        if (typeof coords[0] === 'number') {
          const [x, y] = coords;
          return [
            (x - minX) * scale + offsetX,
            svgHeight - ((y - minY) * scale + offsetY) // Flip Y coordinate
          ];
        } else {
          return coords.map(transformCoordinates);
        }
      };

      // Generate SVG paths
      const paths = geojsonData.features.map((feature, index) => {
        if (!feature.geometry || !feature.geometry.coordinates) return '';
        
        const coords = feature.geometry.coordinates;
        let pathData = '';

        if (feature.geometry.type === 'Polygon') {
          coords.forEach((ring, ringIndex) => {
            const transformedRing = transformCoordinates(ring);
            if (transformedRing.length > 0) {
              pathData += `M ${transformedRing[0][0]},${transformedRing[0][1]} `;
              for (let i = 1; i < transformedRing.length; i++) {
                pathData += `L ${transformedRing[i][0]},${transformedRing[i][1]} `;
              }
              pathData += 'Z ';
            }
          });
        } else if (feature.geometry.type === 'MultiPolygon') {
          coords.forEach(polygon => {
            polygon.forEach(ring => {
              const transformedRing = transformCoordinates(ring);
              if (transformedRing.length > 0) {
                pathData += `M ${transformedRing[0][0]},${transformedRing[0][1]} `;
                for (let i = 1; i < transformedRing.length; i++) {
                  pathData += `L ${transformedRing[i][0]},${transformedRing[i][1]} `;
                }
                pathData += 'Z ';
              }
            });
          });
        }

        return pathData ? `<path d="${pathData}" fill="#FFC107" fill-opacity="0.6" stroke="#FFC107" stroke-width="1"/>` : '';
      }).filter(path => path).join('');

      const svgString = `
        <svg width="${svgWidth}" height="${svgHeight}" viewBox="0 0 ${svgWidth} ${svgHeight}" xmlns="http://www.w3.org/2000/svg">
          <rect width="100%" height="100%" fill="#f8f9fa"/>
          ${paths}
        </svg>
      `;

      // Convert SVG to blob URL
      const blob = new Blob([svgString], { type: 'image/svg+xml' });
      return URL.createObjectURL(blob);
    } catch (error) {
      console.error('Failed to create GeoJSON preview:', error);
      return null;
    }
  };

  // Function to convert TIFF to canvas for preview
  const createTiffPreview = async (file) => {
    try {
      const tiff = await fromBlob(file);
      const image = await tiff.getImage();
      const width = image.getWidth();
      const height = image.getHeight();
      const rasters = await image.readRasters();

      // Create canvas
      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');

      // Create image data
      const imageData = ctx.createImageData(width, height);
      const data = imageData.data;

      // Handle different band configurations
      const numBands = rasters.length;

      for (let i = 0; i < width * height; i++) {
        if (numBands >= 3) {
          // RGB or RGBA
          data[i * 4] = rasters[0][i];     // R
          data[i * 4 + 1] = rasters[1][i]; // G
          data[i * 4 + 2] = rasters[2][i]; // B
          data[i * 4 + 3] = 255;           // A
        } else if (numBands === 1) {
          // Grayscale
          const val = rasters[0][i];
          data[i * 4] = val;
          data[i * 4 + 1] = val;
          data[i * 4 + 2] = val;
          data[i * 4 + 3] = 255;
        }
      }

      ctx.putImageData(imageData, 0, 0);

      // Convert canvas to blob URL
      return new Promise((resolve) => {
        canvas.toBlob((blob) => {
          resolve(URL.createObjectURL(blob));
        });
      });
    } catch (error) {
      console.error('Failed to create TIFF preview:', error);
      return null;
    }
  };

  const handleFileUpload = async (files) => {
    if (files.length > 0) {
      setFileStatus('loading');
      setFileError('');
      setFile(null);
      setPreviewUrl(null);
      setGeojsonData(null);
      setProgress(0);
      const selectedFile = files[0];
      
      // Validate file based on selected type
      const isValidFile = validateFileType(selectedFile, selectedType);
      
      if (!isValidFile) {
        setFileStatus('error');
        setFileError(getFileTypeErrorMessage(selectedType));
        return;
      }
      
      setFile(selectedFile);

      // Auto-fill layer name with filename (without extension)
      const filenameWithoutExt = getFilenameWithoutExtension(selectedFile.name);
      setLayerName(filenameWithoutExt);

      // Handle GeoJSON files
      if (selectedType === 'geojson') {
        try {
          const reader = new FileReader();
          reader.onload = async (e) => {
            try {
              const rawGeojson = JSON.parse(e.target.result);
              setGeojsonData(rawGeojson);
              
              // Create GeoJSON preview
              const geojsonPreviewUrl = createGeojsonPreview(rawGeojson);
              if (geojsonPreviewUrl) {
                setPreviewUrl(geojsonPreviewUrl);
              }
              
              setFileStatus('success');
              setFileError('');
            } catch (parseError) {
              console.error('Failed to parse GeoJSON:', parseError);
              setFileStatus('error');
              setFileError('無効なGeoJSONファイルです。');
            }
          };
          reader.readAsText(selectedFile);
          return;
        } catch (error) {
          console.error('Failed to read GeoJSON file:', error);
          setFileStatus('error');
          setFileError('GeoJSONファイルの読み込みに失敗しました。');
          return;
        }
      }

      // Create preview for image files
      const fileName = selectedFile.name.toLowerCase();
      const isTiff = fileName.match(/\.(tiff|tif)$/);
      const isRegularImage = selectedFile.type.startsWith('image/') && !isTiff;

      if ((isRegularImage || isTiff) && (selectedType === 'orthophoto' || selectedType === 'streetview')) {
        if (isTiff) {
          // Use TIFF preview for .tif/.tiff files
          const tiffPreviewUrl = await createTiffPreview(selectedFile);
          if (tiffPreviewUrl) {
            setPreviewUrl(tiffPreviewUrl);
          }
        } else {
          // Use direct blob URL for regular images
          setPreviewUrl(URL.createObjectURL(selectedFile));
        }
      }
      
      // Just mark file as loaded successfully, processing will happen when Generate button is clicked
      setTimeout(() => {
        setFileStatus('success');
        setFileError('');
      }, 1000); // Simulate loading delay
    }
  };

  // Handle folder upload for dual input (LoD3 data)
  const handleFileUploadDual = async (files, fileType) => {
    if (files.length > 0) {
      // Update file status for this specific input
      setFileStatus(prev => ({ ...prev, [fileType]: 'loading' }));

      // Get folder name from the first file's path
      const firstFile = files[0];
      const folderName = firstFile.webkitRelativePath
        ? firstFile.webkitRelativePath.split('/')[0]
        : `${files.length} files`;

      // Filter files based on type
      let validFiles = files;
      if (fileType === 'pointcloud') {
        // Accept .ply and .las files for pointcloud
        validFiles = files.filter(f =>
          f.name.toLowerCase().endsWith('.ply') || f.name.toLowerCase().endsWith('.las')
        );
      } else if (fileType === 'streetview') {
        // Accept image files for streetview
        validFiles = files.filter(f =>
          f.name.toLowerCase().match(/\.(jpg|jpeg|png|bmp|webp)$/)
        );
      }

      if (validFiles.length === 0) {
        console.error(`No valid files found in folder for ${fileType}`);
        setFileStatus(prev => ({ ...prev, [fileType]: 'error' }));
        return;
      }

      console.log(`📁 Folder "${folderName}" selected for ${fileType}: ${validFiles.length} valid files`);

      // Store files array with folder info
      setFile(prev => ({
        ...prev,
        [fileType]: {
          files: validFiles,
          folderName: folderName,
          fileCount: validFiles.length
        }
      }));

      // Create preview for first image in streetview folder
      if (fileType === 'streetview' && validFiles.length > 0) {
        const previewUrl = URL.createObjectURL(validFiles[0]);
        setPreviewUrl(prev => ({ ...prev, [fileType]: previewUrl }));
      }

      // Mark as success
      setTimeout(() => {
        setFileStatus(prev => ({ ...prev, [fileType]: 'success' }));
      }, 500);
    }
  };

  // Handle drop for dual input
  const handleDropDual = (e, fileType) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    handleFileUploadDual(files, fileType);
  };

  const validateFileType = (file, type) => {
    const fileName = file.name.toLowerCase();
    const fileType = file.type.toLowerCase();
    
    switch (type) {
      case 'geojson':
        return fileName.endsWith('.geojson') || fileName.endsWith('.json') || fileType === 'application/json';
      case 'orthophoto':
        return fileType.startsWith('image/') || fileName.match(/\.(jpg|jpeg|png|tiff|tif)$/);
      case 'pointcloud':
        // return fileName.match(/\.(ply|las|laz|pcd|xyz)$/); // Original format validation - commented for demo
        return true; // Allow any file for demo
      case 'streetview':
        return fileType.startsWith('image/') || fileName.match(/\.(jpg|jpeg|png)$/);
      default:
        return false;
    }
  };

  const getFileTypeErrorMessage = (type) => {
    switch (type) {
      case 'geojson':
        return '有効なGeoJSONファイル（.geojson、.json）を選択してください。';
      case 'orthophoto':
        return '有効な画像ファイル（.jpg、.png、.tiff）を選択してください。';
      case 'pointcloud':
        // return 'Please select a valid point cloud file (.ply, .las, .pcd).'; // Original validation - commented for demo
        return 'デモ用途でどのファイルでも受け入れます。';
      case 'streetview':
        return '有効な画像ファイル（.png、.jpg）を選択してください。';
      default:
        return '無効なファイルタイプです。';
    }
  };

  const getAcceptedFileTypes = (type) => {
    switch (type) {
      case 'geojson':
        return '.geojson,.json,application/json';
      case 'orthophoto':
        return 'image/*,.tiff,.tif';
      case 'pointcloud':
        // return '.ply,.las,.laz,.pcd,.xyz'; // Original formats - commented for demo
        return '*'; // Accept any file for demo
      case 'streetview':
        return '.png,.jpg,.jpeg,image/png,image/jpeg';
      default:
        return '*';
    }
  };

  const handleTypeSelect = (type) => {
    setSelectedType(type);
    setShowTypeSelection(false);
    
    // Handle demo TIFF - show the import panel instead of immediate import
    if (type === 'demo-tiff') {
      // Set up the demo state to show the import panel
      setLayerName('Demo Orthophoto');
      setFileStatus('success');
      
      // Create a fake file object for the demo
      const demoFile = new File(['demo'], 'demo_orthophoto.tif', { type: 'image/tiff' });
      setFile(demoFile);
      
      // Set preview URL for demo image
      setPreviewUrl('/src/assets/result_demo/tiff/route1.tif');
    }
    
    // For LoD3 data, initialize dual file state
    if (type === 'lod3-data') {
      setFile({ pointcloud: null, streetview: null });
      setFileStatus({ pointcloud: 'idle', streetview: 'idle' });
      setPreviewUrl({ pointcloud: null, streetview: null });
    }
  };

  const handleBack = () => {
    if (showLoDSelection) {
      setShowLoDSelection(false);
      setShowTypeSelection(false);
      setSelectedLoD(null);
      setLoDConfig({ height: '', roofType: 'flat' });
    } else {
      setShowTypeSelection(true);
      setSelectedType(null);
      setFile(null);
      setFileStatus('idle');
      setFileError('');
      setPreviewUrl(null);
      setLayerName('');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    handleFileUpload(files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleNext = async () => {
    console.log('🚀 handleNext called with:', {
      showLoDSelection,
      selectedLoD,
      lodConfigHeight: lodConfig.height,
      useBackend,
      backendAvailable
    });
    
    if (showLoDSelection && selectedLoD && lodConfig.height) {
      console.log('✅ LoD selection and height validation passed');
      
      // Check if this is LoD2 generation and backend is available
      if (selectedLoD === 'lod2' && useBackend && backendAvailable) {
        console.log('🎯 Starting backend LoD2 generation...');
        try {
          setFileStatus('backend-processing');
          setProcessingStatus('Generating LoD2 model with backend...');
          setIsProcessing(true);

          // Reset ETA tracking for building phase
          buildingPhaseStartTime.current = null;
          buildingPhaseStartProgress.current = null;
          estimatedTotalBuildings.current = null;

          // Find the current GeoJSON job ID from previously uploaded files
          // This assumes that there's a GeoJSON file already uploaded
          let geojsonJobId = null;
          
          // Try to find existing GeoJSON job ID from backend
          try {
            const jobs = await backendApi.listJobs();
            const geojsonJob = jobs.find(job => job.file_type === 'geojson' && job.completed_at);
            if (geojsonJob) {
              geojsonJobId = geojsonJob.id;
              console.log(`Found existing GeoJSON job: ${geojsonJobId}`);
            }
          } catch (error) {
            console.warn('Could not find existing GeoJSON job:', error);
          }
          
          // Call backend LoD2 generation API
          const result = await backendApi.generateLoD2Model({
            height: lodConfig.height,
            roofType: lodConfig.roofType,
            geojsonJobId: geojsonJobId // Pass the GeoJSON job ID if found
          });
          
          console.log('🎯 LoD2 Generation Response:', {
            jobId: result.job_id,
            expectedObjPath: `/outputs/${result.job_id}/Untitled_lod2.obj`,
            fullUrl: `/outputs/${result.job_id}/Untitled_lod2.obj`
          });
          
          setCurrentJobId(result.job_id);
          setProcessingStatus('Processing LoD2 model...');
          
          // Poll for job completion
          await backendApi.pollJobStatus(result.job_id, (status) => {
            const currentProgress = status.progress || 0;
            setProgress(currentProgress);

            // Calculate and display ETA (only during building generation phase 40-90%)
            const eta = calculateBuildingETA(currentProgress);
            const etaText = eta ? ` (残り約${eta})` : '';
            setProcessingStatus(`Processing LoD2: ${currentProgress.toFixed(0)}%${etaText}`);
          });
          
          // Verify the OBJ file is accessible before marking as complete
          console.log('🔍 Verifying OBJ file accessibility...');
          const objExists = await backendApi.testObjFileAccess(result.job_id);
          
          if (!objExists) {
            throw new Error(`Generated OBJ file not accessible at /outputs/${result.job_id}/Untitled_lod2.obj`);
          }
          
          setFileStatus('success');
          setProcessingStatus('LoD2 model generated successfully!');
          setProgress(100);
          setIsProcessing(false);
          
          console.log('🎉 LoD2 Generation Complete:', {
            jobId: result.job_id,
            objPath: `/outputs/${result.job_id}/Untitled_lod2.obj`,
            verified: objExists
          });
          
          // Import the generated LoD2 model
          console.log('📞 About to call onImport with backend LoD2 data:', {
            type: 'lod2-backend-generated',
            jobId: result.job_id,
            objPath: `/outputs/${result.job_id}/Untitled_lod2.obj`,
            onImportExists: !!onImport
          });
          
          onImport && onImport({ 
            type: 'lod2-backend-generated',
            lodType: selectedLoD,
            config: lodConfig,
            sourceType: selectedType,
            backend: {
              jobId: result.job_id,
              objPath: `/outputs/${result.job_id}/Untitled_lod2.obj`,
              processed: true,
              verified: objExists
            }
          });
          
          console.log('✅ Called onImport successfully for backend LoD2');
          
        } catch (error) {
          console.error('Backend LoD2 generation failed:', error);
          setFileStatus('error');
          setFileError(`LoD2 generation failed: ${error.message}`);
          setProcessingStatus('');
          setIsProcessing(false);
          return;
        }
      } else {
        // Original LoD generation for other types or fallback
        console.log('📤 Using fallback LoD generation - conditions not met for backend:', {
          selectedLoD,
          useBackend,
          backendAvailable,
          isLod2: selectedLoD === 'lod2'
        });
        
        onImport && onImport({ 
          type: 'lod-generation',
          lodType: selectedLoD,
          config: lodConfig,
          sourceType: selectedType
        });
      }
    } else if (selectedType === 'demo-tiff' && layerName) {
      // Handle demo TIFF import with demo file path
      const demoTiffPath = '/src/assets/result_demo/tiff/route1.tif';
      onImport && onImport({ 
        type: 'demo-tiff',
        layerName: layerName,
        demoPath: demoTiffPath
      });
    } else if (selectedType === 'lod3-data' && file?.pointcloud && file?.streetview) {
      // Handle LoD3 dual file import - delegate processing to App.jsx
      setFileStatus('backend-processing');
      setProcessingStatus('LoD3データをアップロード中...');
      setIsProcessing(true);
      setProgress(0);

      // Reset ETA tracking for LoD3 building phase
      lod3PhaseStartTime.current = null;
      lod3PhaseStartProgress.current = null;

      // Prepare import data with progress callbacks
      const importData = {
        files: {
          pointcloud: file.pointcloud,
          streetview: file.streetview
        },
        type: 'lod3-data',
        layerName: layerName || 'Combined_LoD3',
        previewUrls: {
          pointcloud: previewUrl?.pointcloud,
          streetview: previewUrl?.streetview
        },
        // Callbacks for App.jsx to update ImportModel's progress
        callbacks: {
          onProgress: (currentProgress, statusText) => {
            setProgress(currentProgress);
            // Calculate and display ETA (only during building generation phase 30-80%)
            const eta = calculateLod3ETA(currentProgress);
            const etaText = eta ? ` (残り約${eta})` : '';
            setProcessingStatus(statusText || `Processing LoD3: ${currentProgress.toFixed(0)}%${etaText}`);
          },
          onComplete: () => {
            setFileStatus('success');
            setProcessingStatus('LoD3モデルの生成が完了しました！');
            setProgress(100);
            setIsProcessing(false);
          },
          onError: (error) => {
            console.error('Backend LoD3 generation failed:', error);
            setFileStatus('error');
            setFileError(`LoD3 generation failed: ${error.message || error}`);
            setProcessingStatus('');
            setIsProcessing(false);
          }
        }
      };

      onImport && onImport(importData);
    } else if (fileStatus === 'success' && layerName && file) {
      // Check if backend processing should be used
      if (useBackend && backendAvailable) {
        try {
          // Start backend processing when Generate button is clicked
          setFileStatus('backend-processing');
          setProcessingStatus('Uploading file to backend...');
          setIsProcessing(true);

          const result = await backendApi.uploadFile(file, selectedType, layerName);
          setCurrentJobId(result.job_id);
          setProcessingStatus('Processing file...');

          // Poll for job completion with ETA for building generation phase (40-90%)
          await backendApi.pollJobStatus(result.job_id, (status) => {
            const currentProgress = status.progress || 0;
            setProgress(currentProgress);

            // Calculate ETA for building generation phase (40-90%)
            const eta = calculateBuildingETA(currentProgress);
            const etaText = eta ? ` (残り約${eta})` : '';
            setProcessingStatus(`${currentProgress.toFixed(0)}%${etaText}`);
          });
          
          setFileStatus('success');
          setProcessingStatus('Processing completed!');
          setProgress(100);
          setIsProcessing(false);
          
          // Import file data with backend processing results
          const importData = { 
            file, 
            layerName, 
            type: selectedType, 
            previewUrl,
            backend: {
              jobId: result.job_id,
              downloadUrl: backendApi.getDownloadUrl(result.job_id),
              processed: true
            }
          };
          
          onImport && onImport(importData);
          
        } catch (error) {
          console.error('Backend processing failed:', error);
          setFileStatus('error');
          setFileError(`Backend processing failed: ${error.message}`);
          setProcessingStatus('');
          setIsProcessing(false);
        }
      } else {
        // Import file data without backend processing
        const importData = { 
          file, 
          layerName, 
          type: selectedType, 
          previewUrl 
        };
        
        onImport && onImport(importData);
      }
    } else {
      console.log('❌ handleNext conditions not met:', {
        showLoDSelection,
        selectedLoD,
        lodConfigHeight: lodConfig.height,
        selectedType,
        layerName,
        fileStatus,
        file: !!file
      });
    }
  };

  const renderTypeSelection = () => {
    const fileTypes = [
      {
        id: 'lod3-data',
        name: 'MMS点群 & 沿道画像',
        description: 'LoD3モデル生成',
        iconName: ['pointcloud', 'streetView'],
        lodIconName: 'LoD3',
        formats: ['.ply', '.png, .jpg'],
        enabled: projectStage === 'orthophoto-imported' || availableLoDs.has('lod2'),
        disabledReason: projectStage !== 'orthophoto-imported' && !availableLoDs.has('lod2') ? 'LoD2モデルと衛星画像が必要' : null,
        isCombined: true
      },
      {
        id: 'orthophoto',
        name: '衛星画像',
        description: 'LoD2モデル生成',
        iconName: 'orthophoto',
        lodIconName: 'LoD2',
        formats: '.tiff',
        enabled: availableLoDs.has('lod1'),
        disabledReason: !availableLoDs.has('lod1') ? 'LoD1モデルが必要' : null
      },
      // {
      //   id: 'demo-tiff',
      //   name: 'デモ用衛星画像',
      //   description: 'LoD2モデル生成',
      //   iconName: 'orthophoto',
      //   lodIconName: 'LoD2',
      //   formats: '.tiff',
      //   enabled: availableLoDs.has('lod1'),
      //   disabledReason: !availableLoDs.has('lod1') ? 'LoD1モデルが必要' : null
      // },
      {
        id: 'geojson',
        name: '建物フットプリント',
        description: 'LoD1モデル生成',
        iconName: 'geojson',
        lodIconName: 'LoD1',
        formats: '.geojson',
        enabled: projectStage === 'initial',
        disabledReason: projectStage !== 'initial' ? 'GeoJSONはインポート済み' : null
      }
    ];

    return (
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'flex-start', 
        flex: 1, 
        padding: '10px 0 12px 0', 
        width: '100%',
        minHeight: 0, // Allow shrinking
        overflow: 'hidden' // Prevent overflow from this container
      }}>
        {/* Scrollable container for the buttons */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          width: '100%',
          flex: 1,
          minHeight: 0, // Allow shrinking
          overflow: 'auto', // Enable scrolling
          paddingBottom: '20px', // Space before the fixed text
        }}>
          <div style={{
            display: 'flex',
            flexDirection: 'row',
            flexWrap: 'wrap',
            gap: '20px',
            width: '100%',
            maxWidth: '480px',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '0 20px', // Add horizontal padding for scrolling
            boxSizing: 'border-box'
          }}>
            {fileTypes.map((type) => (
              <div
                key={type.id}
                onClick={() => type.enabled && handleTypeSelect(type.id)}
                className="file-type-card"
                style={{
                  border: type.enabled ? '2px solid #e0e0e0' : '2px solid #ccc',
                  borderRadius: '12px',
                  padding: '18px 24px',
                  cursor: type.enabled ? 'pointer' : 'not-allowed',
                  textAlign: 'left',
                  transition: 'all 0.3s ease',
                  backgroundColor: type.enabled ? '#fff' : '#f5f5f5',
                  opacity: type.enabled ? 1 : 0.6,
                  display: 'flex',
                  alignItems: 'center',
                  minHeight: type.isCombined ? '160px' : '120px',
                  boxShadow: type.enabled ? '0 2px 8px rgba(140,43,226,0.04)' : 'none',
                  width: '100%',
                  maxWidth: '440px',
                  flex: '1 1 320px',
                  margin: 0,
                  flexShrink: 0 // Prevent buttons from shrinking
                }}
              >
                {/* Icon group on the left */}
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minWidth: 120,
                  marginRight: 40,
                }}>
                  {type.isCombined ? (
                    // Combined icon display for LoD3 data
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        {/* Left side: Two data icons stacked */}
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                          <Icon
                            name={type.iconName[0]}
                            size={64}
                            noFilter={!type.enabled}
                            style={{ filter: type.enabled ? 'none' : 'grayscale(100%)' }}
                          />
                          <Icon
                            name={type.iconName[1]}
                            size={64}
                            noFilter={!type.enabled}
                            style={{ filter: type.enabled ? 'none' : 'grayscale(100%)' }}
                          />
                        </div>
                        {/* Middle: Arrow (centered vertically) */}
                        <span style={{ fontSize: 32, color: type.enabled ? '#8A2BE2' : '#ccc', fontWeight: 'bold', margin: '0 8px' }}>→</span>
                        {/* Right side: LoD3 icon (centered vertically) */}
                        <Icon
                          name={type.lodIconName}
                          size={64}
                          noFilter={!type.enabled}
                          style={{ filter: type.enabled ? 'none' : 'grayscale(100%)', stroke: '#7b6ae9' }}
                        />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginTop: 6 }}>
                        <div style={{ fontSize: 12, color: type.enabled ? '#666' : '#999', fontWeight: 500, textAlign: 'center', width: 64 }}>データ</div>
                        <div style={{ fontSize: 12, color: type.enabled ? '#666' : '#999', fontWeight: 500, textAlign: 'center', width: 64 }}>{type.lodIconName}</div>
                      </div>
                    </>
                  ) : (
                    // Original single icon display
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                        <Icon
                          name={type.iconName}
                          size={64}
                          noFilter={!type.enabled}
                          style={{ filter: type.enabled ? 'none' : 'grayscale(100%)', stroke: (['LoD1','LoD2','LoD3'].includes(type.lodIconName) ? '#7b6ae9' : undefined) }}
                        />
                        <span style={{ fontSize: 32, color: type.enabled ? '#8A2BE2' : '#ccc', fontWeight: 'bold', margin: '0 8px' }}>→</span>
                        <Icon
                          name={type.lodIconName}
                          size={64}
                          noFilter={!type.enabled}
                          style={{ filter: type.enabled ? 'none' : 'grayscale(100%)', stroke: (['LoD1','LoD2','LoD3'].includes(type.lodIconName) ? '#7b6ae9' : undefined) }}
                        />
                      </div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginTop: 6 }}>
                        <div style={{ fontSize: 12, color: type.enabled ? '#666' : '#999', fontWeight: 500, textAlign: 'center', width: 64 }}>データ</div>
                        <div style={{ fontSize: 12, color: type.enabled ? '#666' : '#999', fontWeight: 500, textAlign: 'center', width: 64 }}>{type.lodIconName}</div>
                      </div>
                    </>
                  )}
                </div>
                {/* Text group on the right */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: 4, color: type.enabled ? '#333' : '#999' }}>
                    {type.name}
                  </div>
                  <div style={{ fontSize: '14px', color: type.enabled ? '#666' : '#999', marginBottom: 4, lineHeight: '1.4' }}>
                    {type.description}
                  </div>
                  <div style={{ fontSize: '12px', color: type.enabled ? '#888' : '#aaa', fontStyle: 'italic', marginBottom: 2 }}>
                    {type.isCombined ? (
                      <>
                        <div>MMS点群: {type.formats[0]}</div>
                        <div>沿道画像: {type.formats[1]}</div>
                      </>
                    ) : (
                      `フォーマット: ${type.formats}`
                    )}
                  </div>
                  {!type.enabled && type.disabledReason && (
                    <div style={{ fontSize: '12px', color: '#8A2BE2', marginTop: '4px', fontWeight: '500' }}>
                      {type.disabledReason}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Fixed instruction text at the bottom */}
        <div style={{ 
          fontSize: '18px', 
          color: '#333', 
          textAlign: 'center', 
          width: '100%',
          padding: '16px 20px 0 20px',
          borderTop: '1px solid #f0f0f0', // Subtle separator
          flexShrink: 0 // Prevent this text from being compressed
        }}>
          データタイプを選択してください
        </div>
      </div>
    );
  };

  const renderLoDSelection = () => {
    const lodTypes = [
      {
        id: 'lod1',
        name: 'LoD 1',
        description: '基本的な幾何学的表現を持つブロックモデル',
        iconName: 'LoD1',
        enabled: selectedType === 'geojson', // Only LoD1 for GeoJSON
        configFields: ['height']
      },
      {
        id: 'lod2',
        name: 'LoD 2',
        description: '屋根構造を持つ詳細モデル',
        iconName: 'LoD2',
        enabled: selectedType === 'orthophoto', // Only LoD2 for orthophoto
        configFields: ['height', 'roofType']
      },
      {
        id: 'lod3',
        name: 'LoD 3',
        description: '建築的特徴を持つ高詳細モデル',
        iconName: 'LoD3',
        enabled: selectedType === 'pointcloud' || selectedType === 'streetview', // LoD3 for point cloud and streetview
        configFields: ['height', 'roofType', 'wallTexture']
      }
    ];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'stretch', justifyContent: 'flex-start', flex: 1, padding: '20px 0', width: '100%' }}>
        <div style={{ fontSize: '18px', marginBottom: '32px', color: '#333', textAlign: 'left' }}>
          生成する詳細レベル（LoD）を選択してください：
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', width: '100%' }}>
          {lodTypes.map((lod) => (
            <div
              key={lod.id}
              onClick={() => lod.enabled && setSelectedLoD(lod.id)}
              className="file-type-card"
              style={{
                border: selectedLoD === lod.id ? '3px solid #8A2BE2' : (lod.enabled ? '2px solid #e0e0e0' : '2px solid #ccc'),
                borderRadius: '12px',
                padding: '24px',
                cursor: lod.enabled ? 'pointer' : 'not-allowed',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                backgroundColor: selectedLoD === lod.id ? '#f8f4ff' : (lod.enabled ? '#fff' : '#f5f5f5'),
                opacity: lod.enabled ? 1 : 0.6
              }}
            >
              <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'center' }}>
                <Icon 
                  name={lod.iconName} 
                  size={96} 
                  noFilter={lod.enabled}
                  style={{ filter: lod.enabled ? 'none' : 'grayscale(100%)', stroke: (['LoD1','LoD2','LoD3'].includes(lod.iconName) ? '#7b6ae9' : undefined) }}
                />
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '8px', color: lod.enabled ? '#333' : '#999' }}>
                {lod.name}
              </div>
              <div style={{ fontSize: '14px', color: lod.enabled ? '#666' : '#999', marginBottom: '12px', lineHeight: '1.4' }}>
                {lod.description}
              </div>
              {!lod.enabled && (
                <div style={{ fontSize: '12px', color: '#8A2BE2', marginTop: '8px', fontWeight: '500' }}>
                  {selectedType === 'geojson' && lod.id !== 'lod1' ? 'GeoJSONにはLoD1のみ利用可能' :
                   selectedType === 'orthophoto' && lod.id !== 'lod2' ? 'オルソフォトにはLoD2のみ利用可能' :
                   (selectedType === 'pointcloud' || selectedType === 'streetview') && lod.id !== 'lod3' ? 'ポイントクラウド・ストリートビューにはLoD3のみ利用可能' :
                   'このデータタイプでは利用できません'}
                </div>
              )}
            </div>
          ))}
        </div>
        
        {selectedLoD && (
          <div style={{ marginTop: '32px', padding: '20px', border: '1px solid #e0e0e0', borderRadius: '8px', width: '100%' }}>
            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '16px', color: '#333' }}>
              {selectedLoD.toUpperCase()}の設定
            </div>
            
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#555' }}>
                高さ（メートル）：
              </label>
              <input
                type="number"
                value={lodConfig.height}
                onChange={(e) => setLoDConfig(prev => ({ ...prev, height: e.target.value }))}
                placeholder="建物の高さを入力"
                style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
              />
            </div>
            
            {(selectedLoD === 'lod2' || selectedLoD === 'lod3') && (
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#555' }}>
                  屋根タイプ：
                </label>
                <select
                  value={lodConfig.roofType}
                  onChange={(e) => setLoDConfig(prev => ({ ...prev, roofType: e.target.value }))}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="flat">陸屋根</option>
                  <option value="stepped-flat">段差あり屋根</option>
                  <option value="folded">混合屋根</option>
                  <option value="gable">切妻屋根</option>
                  <option value="hipped">寄棟屋根</option>
                </select>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderPreviewSection = () => {
    if (!selectedType) return null;

    // Dual preview for LoD3 data (pointcloud + streetview)
    if (selectedType === 'lod3-data') {
      return (
        <div className="preview-container" style={{ flex: '1 1 0', display: 'flex', flexDirection: 'row', gap: '16px', justifyContent: 'space-between', alignItems: 'stretch', marginBottom: '0', minHeight: '200px', width: '100%' }}>
          {/* Left panel: MMS Point Cloud */}
          <div className="preview-box dual" style={{ position: 'relative', flex: '1 1 50%', minHeight: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '2px dashed #cccccc', borderRadius: '8px', background: 'rgba(255,255,255,0.5)', boxSizing: 'border-box' }}>
            <div className="preview-title" style={{ fontWeight: 'bold', marginBottom: '16px', textAlign: 'center', zIndex: 2 }}>
              MMS点群
            </div>
            {file?.pointcloud && previewUrl?.pointcloud ? (
              <img src={previewUrl.pointcloud} alt="Point cloud preview" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px' }} />
            ) : (
              <div
                className="preview-placeholder"
                onDrop={(e) => handleDropDual(e, 'pointcloud')}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.pointcloud?.click()}
                style={{
                  border: dragOver ? '2px dashed #8A2BE2' : '2px dashed #cccccc',
                  backgroundColor: dragOver ? 'rgba(138, 43, 226, 0.1)' : 'transparent',
                  width: '90%',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto',
                  fontSize: '14px',
                  color: '#666',
                  cursor: 'pointer',
                  zIndex: 2,
                }}
              >
                {fileStatus?.pointcloud === 'loading' && (
                  <div className="preview-status loading">読み込み中...</div>
                )}
                {fileStatus?.pointcloud === 'success' && file?.pointcloud && (
                  <div className="preview-status success">
                    読み込み完了: {file.pointcloud.folderName} ({file.pointcloud.fileCount}個の点群ファイル)
                  </div>
                )}
                {fileStatus?.pointcloud === 'error' && (
                  <div className="preview-status error">エラー</div>
                )}
                {fileStatus?.pointcloud === 'idle' && !dragOver && (
                  <div className="preview-placeholder-text">フォルダを選択 (PLY/LASファイル)</div>
                )}
              </div>
            )}
            <input
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              ref={(el) => {
                if (!fileInputRef.current) fileInputRef.current = {};
                fileInputRef.current.pointcloud = el;
              }}
              style={{ display: 'none' }}
              onChange={(e) => handleFileUploadDual(Array.from(e.target.files), 'pointcloud')}
            />
          </div>

          {/* Right panel: Street View */}
          <div className="preview-box dual" style={{ position: 'relative', flex: '1 1 50%', minHeight: '200px', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '2px dashed #cccccc', borderRadius: '8px', background: 'rgba(255,255,255,0.5)', boxSizing: 'border-box' }}>
            <div className="preview-title" style={{ fontWeight: 'bold', marginBottom: '16px', textAlign: 'center', zIndex: 2 }}>
              沿道画像
            </div>
            {file?.streetview && previewUrl?.streetview ? (
              <img src={previewUrl.streetview} alt="Street view preview" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px' }} />
            ) : (
              <div
                className="preview-placeholder"
                onDrop={(e) => handleDropDual(e, 'streetview')}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current?.streetview?.click()}
                style={{
                  border: dragOver ? '2px dashed #8A2BE2' : '2px dashed #cccccc',
                  backgroundColor: dragOver ? 'rgba(138, 43, 226, 0.1)' : 'transparent',
                  width: '90%',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto',
                  fontSize: '14px',
                  color: '#666',
                  cursor: 'pointer',
                  zIndex: 2,
                }}
              >
                {fileStatus?.streetview === 'loading' && (
                  <div className="preview-status loading">読み込み中...</div>
                )}
                {fileStatus?.streetview === 'success' && file?.streetview && (
                  <div className="preview-status success">
                    読み込み完了: {file.streetview.folderName} ({file.streetview.fileCount}個の画像)
                  </div>
                )}
                {fileStatus?.streetview === 'error' && (
                  <div className="preview-status error">エラー</div>
                )}
                {fileStatus?.streetview === 'idle' && !dragOver && (
                  <div className="preview-placeholder-text">フォルダを選択 (画像ファイル)</div>
                )}
              </div>
            )}
            <input
              type="file"
              webkitdirectory=""
              directory=""
              multiple
              ref={(el) => {
                if (!fileInputRef.current) fileInputRef.current = {};
                fileInputRef.current.streetview = el;
              }}
              style={{ display: 'none' }}
              onChange={(e) => handleFileUploadDual(Array.from(e.target.files), 'streetview')}
            />
          </div>
        </div>
      );
    }

    // Original single preview for other types
    const getPlaceholderText = (type) => {
      switch (type) {
        case 'geojson': return 'GeoJSONファイルをここにドロップするか、クリックしてアップロード';
        case 'orthophoto': return '衛星画像をここにドロップするか、クリックしてアップロード';
        case 'demo-tiff': return 'デモオルソフォトが処理の準備完了';
        case 'pointcloud': return 'ポイントクラウドファイルをここにドロップするか、クリックしてアップロード';
        case 'streetview': return 'ストリートビュー画像をここにドロップするか、クリックしてアップロード';
        default: return 'ファイルをここにドロップするか、クリックしてアップロード';
      }
    };

    return (
      <div className="preview-container" style={{ flex: '1 1 0', display: 'flex', flexDirection: 'column', justifyContent: 'flex-start', alignItems: 'stretch', marginBottom: '0', minHeight: '200px', width: '100%' }}>
        <div className="preview-box single" style={{ position: 'relative', flex: '1 1 0', minHeight: '200px', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '2px dashed #cccccc', borderRadius: '8px', background: 'rgba(255,255,255,0.5)', boxSizing: 'border-box' }}>
          {previewUrl && (selectedType === 'orthophoto' || selectedType === 'demo-tiff' || selectedType === 'streetview') ? (
            <img src={previewUrl} alt="Orthophoto preview" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px' }} />
          ) : previewUrl && selectedType === 'geojson' ? (
            <img src={previewUrl} alt="GeoJSON preview" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px' }} />
          ) : selectedType === 'demo-tiff' ? (
            <div style={{ textAlign: 'center', zIndex: 2 }}>
              <div className="preview-title" style={{ fontWeight: 'bold', marginBottom: '16px' }}>
                プレビュー
              </div>
              <div style={{ fontSize: '16px', color: '#666' }}>
                デモ用衛星画像読込完了
              </div>
            </div>
          ) : (
            <>
              <div className="preview-title" style={{ fontWeight: 'bold', marginBottom: '16px', textAlign: 'center', zIndex: 2 }}>
                {/* {getTypeDisplayName(selectedType)}の */}
                プレビュー
              </div>
              <div
                className="preview-placeholder"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
                style={{
                  border: dragOver ? '2px dashed #8A2BE2' : '2px dashed #cccccc',
                  backgroundColor: dragOver ? 'rgba(138, 43, 226, 0.1)' : 'transparent',
                  width: '320px',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto',
                  fontSize: '16px',
                  color: '#666',
                  marginBottom: '0',
                  cursor: 'pointer',
                  zIndex: 2,
                }}
              >
                {fileStatus === 'loading' && (
                  <div className="preview-status loading">読み込み中...</div>
                )}
                {fileStatus === 'backend-processing' && (
                  <div className="preview-status loading">
                    {processingStatus || 'バックエンド処理中...'}
                  </div>
                )}
                {fileStatus === 'success' && file && (
                  <div className="preview-status success">
                    {useBackend && backendAvailable ? '処理完了' : 'ファイル読み込み完了'}: {file.name}
                  </div>
                )}
                {fileStatus === 'error' && fileError && (
                  <div className="preview-status error">{fileError}</div>
                )}
                {fileStatus === 'idle' && !dragOver && (
                  <div className="preview-placeholder-text">{getPlaceholderText(selectedType)}</div>
                )}
                <input
                  type="file"
                  accept={getAcceptedFileTypes(selectedType)}
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={e => handleFileUpload(Array.from(e.target.files))}
                />
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="modal-overlay">
      <div
        className="import-modal"
        style={showTypeSelection
          ? {
              display: 'flex', flexDirection: 'column', maxHeight: '90vh', height: 'auto', minHeight: '0', width: '100%', maxWidth: '520px', padding: '32px 0 24px 0', boxSizing: 'border-box', justifyContent: 'flex-start', alignItems: 'center', margin: '0 auto'
            }
          : {
              display: 'flex', flexDirection: 'column', maxHeight: '80vh', height: '100%', padding: '40px', boxSizing: 'border-box', justifyContent: 'flex-start', alignItems: 'stretch', margin: '0 auto', width: '800px', minWidth: '600px'
            }
        }
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '24px', width: '100%' }}>
          {!showTypeSelection && (
            <button
              onClick={handleBack}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '18px',
                cursor: 'pointer',
                marginRight: '16px',
                color: '#8A2BE2',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              ← 戻る
            </button>
          )}
          <div className="import-modal-title" style={{ margin: '0 0 0 36px' }}>
            {showTypeSelection ? 'インポート' : 
             showLoDSelection ? `${selectedType === 'pointcloud' ? 'MMS点群' : selectedType === 'demo-tiff' ? '衛星画像' : selectedType === 'geojson' ? '建物フットプリント' : selectedType?.toUpperCase()} LoDを設定` :
             `${selectedType === 'pointcloud' ? 'MMS点群' : selectedType === 'orthophoto' ? '衛星画像' : selectedType === 'geojson' ? '建物フットプリント' : selectedType?.toUpperCase()}をインポート`}
          </div>
        </div>
        
        {showTypeSelection ? renderTypeSelection() : 
         showLoDSelection ? (
          <>
            {renderLoDSelection()}
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '24px', gap: '16px' }}>
              <button
                className="model-button primary"
                disabled={!selectedLoD || !lodConfig.height}
                style={{ 
                  minWidth: '120px', 
                  height: '40px', 
                  fontSize: '22px', 
                  background: (!selectedLoD || !lodConfig.height) ? '#ccc' : '#8BC34A', 
                  color: '#fff', 
                  borderRadius: '8px', 
                  fontWeight: 'bold', 
                  border: 'none', 
                  cursor: (!selectedLoD || !lodConfig.height) ? 'not-allowed' : 'pointer',
                  transition: 'all 0.15s ease',
                  transform: 'scale(1)'
                }}
                onClick={handleNext}
                onMouseDown={(e) => {
                  if (!(!selectedLoD || !lodConfig.height)) {
                    e.target.style.background = '#7BA946';
                    e.target.style.transform = 'scale(0.98)';
                  }
                }}
                onMouseUp={(e) => {
                  if (!(!selectedLoD || !lodConfig.height)) {
                    e.target.style.background = '#8BC34A';
                    e.target.style.transform = 'scale(1)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!(!selectedLoD || !lodConfig.height)) {
                    e.target.style.background = '#8BC34A';
                    e.target.style.transform = 'scale(1)';
                  }
                }}
              >
                {selectedLoD?.toUpperCase()}を生成
              </button>
            </div>
          </>
         ) : (
          <>
            {renderPreviewSection()}
            <div style={{ display: 'flex', alignItems: 'center', width: '100%', marginTop: '24px', marginBottom: '8px', gap: '16px' }}>
              {selectedType !== 'lod3-data' && (
                <div style={{ flex: 2, display: 'flex', alignItems: 'center' }}>
                  <label style={{ fontWeight: '500', fontSize: '16px', color: '#222', marginRight: '8px', whiteSpace: 'nowrap' }}>
                    {/* {selectedType === 'geojson' ? '建物フットプリント' : 
                     selectedType === 'orthophoto' ? 'オルソフォト' : 
                     selectedType === 'demo-tiff' ? '衛星画像' : 
                     'MMS点群'} */}
                     レイヤー名
                  </label>
                  <input
                    type="text"
                    className="import-input"
                    value={layerName}
                    onChange={(e) => setLayerName(e.target.value)}
                    placeholder=""
                    style={{ width: '180px', height: '28px', fontSize: '16px', marginRight: '8px' }}
                  />
                </div>
              )}
              <div style={{ 
                flex: 1, 
                textAlign: selectedType === 'lod3-data' ? 'left' : 'center', 
                color: '#222', 
                fontSize: '16px', 
                minHeight: 24, 
                display: 'flex', 
                flexDirection: 'column', 
                alignItems: selectedType === 'lod3-data' ? 'flex-start' : 'center', 
                justifyContent: 'center' 
              }}>
                {selectedType === 'lod3-data' ? (
                  // Status for dual file upload
                  <>
                    {fileStatus === 'backend-processing' && (
                      <div>{processingStatus || '処理中...'}</div>
                    )}
                    {fileStatus === 'success' && !isProcessing && '生成完了！'}
                    {fileStatus !== 'backend-processing' && fileStatus !== 'success' && (
                      <>
                        {fileStatus?.pointcloud === 'idle' && fileStatus?.streetview === 'idle' && '入力待ち'}
                        {(fileStatus?.pointcloud === 'loading' || fileStatus?.streetview === 'loading') && '読み込み中...'}
                        {fileStatus?.pointcloud === 'success' && fileStatus?.streetview === 'success' && '読み込み完了！'}
                        {fileStatus?.pointcloud === 'success' && fileStatus?.streetview === 'idle' && 'MMS点群読み込み完了'}
                        {fileStatus?.pointcloud === 'idle' && fileStatus?.streetview === 'success' && '沿道画像読み込み完了'}
                      </>
                    )}
                  </>
                ) : (
                  // Status for single file upload
                  <>
                    {fileStatus === 'idle' && '入力待ち'}
                    {fileStatus === 'loading' && '読み込み中...'}
                    {fileStatus === 'backend-processing' && (
                      <>
                        <div>{processingStatus || '処理中...'}</div>
                      </>
                    )}
                    {fileStatus === 'success' && !isProcessing && '読み込み完了！'}
                    {fileStatus === 'error' && fileError}
                  </>
                )}
              </div>
              <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  className="model-button primary"
                  disabled={
                    selectedType === 'lod3-data' 
                      ? (fileStatus?.pointcloud !== 'success' || fileStatus?.streetview !== 'success' || isProcessing)
                      : (!layerName || (fileStatus !== 'success' && fileStatus !== 'backend-processing') || isProcessing)
                  }
                  style={{ 
                    minWidth: '140px', 
                    height: '40px', 
                    fontSize: '22px', 
                    background: '#8BC34A', 
                    color: '#fff', 
                    borderRadius: '8px', 
                    fontWeight: 'bold', 
                    border: 'none', 
                    cursor: (selectedType === 'lod3-data' 
                      ? (fileStatus?.pointcloud !== 'success' || fileStatus?.streetview !== 'success' || isProcessing)
                      : (!layerName || (fileStatus !== 'success' && fileStatus !== 'backend-processing') || isProcessing)) ? 'not-allowed' : 'pointer',
                    transition: 'all 0.15s ease',
                    transform: 'scale(1)'
                  }}
                  onClick={handleNext}
                  onMouseDown={(e) => {
                    const isDisabled = selectedType === 'lod3-data' 
                      ? (fileStatus?.pointcloud !== 'success' || fileStatus?.streetview !== 'success' || isProcessing)
                      : (!layerName || (fileStatus !== 'success' && fileStatus !== 'backend-processing') || isProcessing);
                    if (!isDisabled) {
                      e.target.style.background = '#7BA946';
                      e.target.style.transform = 'scale(0.98)';
                    }
                  }}
                  onMouseUp={(e) => {
                    const isDisabled = selectedType === 'lod3-data' 
                      ? (fileStatus?.pointcloud !== 'success' || fileStatus?.streetview !== 'success' || isProcessing)
                      : (!layerName || (fileStatus !== 'success' && fileStatus !== 'backend-processing') || isProcessing);
                    if (!isDisabled) {
                      e.target.style.background = '#8BC34A';
                      e.target.style.transform = 'scale(1)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    const isDisabled = selectedType === 'lod3-data' 
                      ? (fileStatus?.pointcloud !== 'success' || fileStatus?.streetview !== 'success' || isProcessing)
                      : (!layerName || (fileStatus !== 'success' && fileStatus !== 'backend-processing') || isProcessing);
                    if (!isDisabled) {
                      e.target.style.background = '#8BC34A';
                      e.target.style.transform = 'scale(1)';
                    }
                  }}
                >
                  {isProcessing ? '処理中...' : '生成'}
                </button>
              </div>
            </div>
            <div style={{ width: '100%', height: '8px', background: '#eee', borderRadius: '4px', marginTop: '0', marginBottom: '0', overflow: 'hidden' }}>
              <div style={{
                width: selectedType === 'lod3-data'
                  ? (fileStatus === 'backend-processing' ? `${progress}%` :
                     fileStatus === 'success' ? '100%' :
                     fileStatus?.pointcloud === 'success' && fileStatus?.streetview === 'success' ? '100%' :
                     fileStatus?.pointcloud === 'success' || fileStatus?.streetview === 'success' ? '50%' : '0%')
                  : (fileStatus === 'loading' ? '50%' :
                     fileStatus === 'backend-processing' ? `${progress}%` :
                     fileStatus === 'success' ? '100%' : '0%'),
                height: '100%',
                background: (selectedType === 'lod3-data'
                  ? (fileStatus === 'success' ? '#8BC34A' :
                     fileStatus === 'backend-processing' ? '#8A2BE2' :
                     fileStatus?.pointcloud === 'success' && fileStatus?.streetview === 'success' ? '#8BC34A' : '#8A2BE2')
                  : (fileStatus === 'success' ? '#8BC34A' : '#8A2BE2')),
                transition: 'width 0.5s' 
              }} />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Only re-render if isOpen changes or critical props change
  if (prevProps.isOpen !== nextProps.isOpen) return false;
  if (prevProps.projectStage !== nextProps.projectStage) return false;
  if (prevProps.availableLoDs !== nextProps.availableLoDs) return false;

  // Compare callbacks
  if (prevProps.onClose !== nextProps.onClose) return false;
  if (prevProps.onImport !== nextProps.onImport) return false;

  // If modal is closed, don't re-render on any prop changes
  if (!nextProps.isOpen && !prevProps.isOpen) return true;

  return true;
};

export default React.memo(ImportModel, arePropsEqual);