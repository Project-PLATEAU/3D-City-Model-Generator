import { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import { fromArrayBuffer } from 'geotiff';
import proj4 from 'proj4';
import { updatePolygonHeight, validateGeojsonData, getHeightStatistics } from './utils/geojsonUtils';
import { generateUniqueId, getDisplayName, createUnifiedLayer, LayerManager, createLayerControl, generateControlId, extractFileMetadata, generateLoDModelName } from './utils/layerUtils';
import backendApi from './services/backendApi';
// import footprintGeojson from './assets/test_data/01/footprint_1.geojson';

// Components
import MapContainer from './components/MapContainer';
import Toolbar from './components/Toolbar';
import LayerPanel from './components/LayerPanel';
import OrthophotoPreview from './components/OrthophotoPreview';
import ImportModel from './components/ImportModel';
import ConfigurationPanel from './components/ConfigurationPanel';
// import GeojsonDebugPanel from './components/GeojsonDebugPanel';
import LoDButtons from './components/LoDButtons';
import ObjModelViewer from './components/ObjModelViewer';
import { Icon } from './utils/iconLoader';

function App() {
  // Unified Layer Management System
  const layerManagerRef = useRef(new LayerManager());
  const [allLayers, setAllLayers] = useState([]); // Unified layer state
  const [layerUpdateTrigger, setLayerUpdateTrigger] = useState(0); // Force re-render when layers change

  // State management
  const [activeTool, setActiveTool] = useState('hand');
  const [mapInstance, setMapInstance] = useState(null);
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [mapState, setMapState] = useState('normal'); // normal, configuration, loading
  const [showImportModel, setShowImportModel] = useState(false);
  const [visibleLayers, setVisibleLayers] = useState(new Set([])); // Start with all layers unchecked
  const [showLayerPanel, setShowLayerPanel] = useState(false);
  // Legacy states for backward compatibility (will be gradually replaced)
  const [customLayers, setCustomLayers] = useState([]);
  const [showGeojson, setShowGeojson] = useState(false);
  const [geojsonData, setGeojsonData] = useState(null);
  const [importedGeojsonLayers, setImportedGeojsonLayers] = useState([]); // Store imported GeoJSON layers separately
  const [lod1Layers, setLod1Layers] = useState([]); // Store generated LoD1 models separately
  const [lod3Layers, setLod3Layers] = useState([]); // Store LoD3 models separately
  const [currentGeojsonFile, setCurrentGeojsonFile] = useState(null); // Store current GeoJSON file info

  // Helper function to trigger layer updates
  const triggerLayerUpdate = () => {
    setAllLayers([...layerManagerRef.current.getAllLayers()]);
    setLayerUpdateTrigger(prev => prev + 1);
  };

  // Helper function to wait for a layer to be loaded on the map
  const waitForLayerToLoad = (layerId, callback, maxAttempts = 20) => {
    let attempts = 0;
    
    const checkLayer = () => {
      attempts++;
      console.log(`🔍 Checking if layer ${layerId} is loaded... (attempt ${attempts})`);
      
      // For GeoJSON layers, check for the actual layer IDs that MapContainer creates
      const layersToCheck = [];
      
      // Check if the layer exists directly
      if (mapInstance && mapInstance.getLayer && mapInstance.getLayer(layerId)) {
        layersToCheck.push(layerId);
      }
      
      // For GeoJSON layers, also check for the fixed layer IDs that MapContainer creates
      const layer = layerManagerRef.current.getLayer(layerId);
      if (layer && layer.type === 'geojson') {
        // MapContainer creates these fixed layer IDs for GeoJSON
        const geojsonLayerIds = [
          'geojson-mask-extrusion',
          'geojson-mask-fill', 
          'geojson-mask-stroke'
        ];
        
        for (const geojsonLayerId of geojsonLayerIds) {
          if (mapInstance && mapInstance.getLayer && mapInstance.getLayer(geojsonLayerId)) {
            layersToCheck.push(geojsonLayerId);
          }
        }
      }
      
      if (layersToCheck.length > 0) {
        console.log(`✅ Layer ${layerId} is now loaded on map (found: ${layersToCheck.join(', ')})`);
        callback();
        return;
      }
      
      if (attempts < maxAttempts) {
        setTimeout(checkLayer, 100);
      } else {
        console.warn(`⚠️ Layer ${layerId} failed to load after ${maxAttempts} attempts`);
        // Still call the callback to continue with the process even if layer detection failed
        console.log(`🔄 Continuing anyway - the layer might be there but not detected`);
        callback();
      }
    };
    
    checkLayer();
  };

  // Helper function to refresh a layer visibility without affecting toggle state
  const refreshLayer = (layerId, forceVisible = true) => {
    console.log(`🔄 Refreshing layer visibility: ${layerId}`);
    
    // For GeoJSON layers, we need to work with the actual layer IDs that MapContainer creates
    const layersToRefresh = [];
    
    // Check if the layer exists directly
    if (mapInstance && mapInstance.getLayer && mapInstance.getLayer(layerId)) {
      layersToRefresh.push(layerId);
    }
    
    // For GeoJSON layers, also refresh the fixed layer IDs that MapContainer creates
    const layer = layerManagerRef.current.getLayer(layerId);
    if (layer && layer.type === 'geojson') {
      const geojsonLayerIds = [
        'geojson-mask-extrusion',
        'geojson-mask-fill', 
        'geojson-mask-stroke'
      ];
      
      for (const geojsonLayerId of geojsonLayerIds) {
        if (mapInstance && mapInstance.getLayer && mapInstance.getLayer(geojsonLayerId)) {
          layersToRefresh.push(geojsonLayerId);
        }
      }
    }
    
    if (layersToRefresh.length > 0) {
      layersToRefresh.forEach(actualLayerId => {
        try {
          const currentVisibility = mapInstance.getLayoutProperty(actualLayerId, 'visibility');
          console.log(`📊 Current visibility for ${actualLayerId}: ${currentVisibility}`);
          
          if (forceVisible) {
            mapInstance.setLayoutProperty(actualLayerId, 'visibility', 'visible');
            console.log(`✅ Forced layer visible: ${actualLayerId}`);
          } else {
            const newVisibility = currentVisibility === 'visible' ? 'none' : 'visible';
            mapInstance.setLayoutProperty(actualLayerId, 'visibility', newVisibility);
            console.log(`🔄 Toggled ${actualLayerId} visibility to: ${newVisibility}`);
          }
        } catch (error) {
          console.warn(`⚠️ Error refreshing layer ${actualLayerId}:`, error);
        }
      });
    } else {
      console.warn(`⚠️ No layers found to refresh for: ${layerId}`);
    }
  };

  // Expose refresh and wait functions through mapContainerRef for external access
  useEffect(() => {
    if (mapContainerRef.current) {
      mapContainerRef.current.refreshLayer = refreshLayer;
      mapContainerRef.current.waitForLayerToLoad = waitForLayerToLoad;
    }
  }, [mapInstance]);

  // Auto-detect and add existing LoD2 models from backend
  const checkAndAddExistingLoD2Models = async () => {
    try {
      console.log('🔍 Checking for existing LoD2 models...');
      const jobs = await backendApi.listJobs();
      
      // Look for orthophoto jobs that might have generated LoD2 models
      const orthophotoJobs = jobs.filter(job => job.file_type === 'orthophoto' && job.completed_at);
      console.log('📋 Found orthophoto jobs:', orthophotoJobs);
      
      for (const job of orthophotoJobs) {
        console.log('🔍 Processing job:', job);
        
        // Extract job ID - try different possible field names
        const jobId = job.id || job.job_id || job.ID || Object.keys(job).find(key => 
          key.toLowerCase().includes('id') && job[key] && typeof job[key] === 'string'
        );
        console.log('📋 Extracted job ID:', jobId);
        
        if (!jobId) {
          console.warn('⚠️ No job ID found for job:', job);
          continue;
        }
        
        // Construct expected filename using the job's layer_name
        const layerName = job.layer_name || 'Untitled';
        const expectedFileName = `${layerName}_lod2.obj`;
        const expectedFileNameBMQI = `${layerName}_lod2_bmqi.obj`;

        // Check multiple possible LoD2 file names (including expected and fallbacks)
        const possibleLoD2Files = [expectedFileName, 'Untitled_lod2.obj', 'lod2.obj'];

        // First, check for regular LoD2 model
        for (const fileName of possibleLoD2Files) {
          const lod2ObjPath = `/outputs/${jobId}/${fileName}`;
          console.log(`🔍 Checking for LoD2 file: ${lod2ObjPath}`);

          const objExists = await backendApi.testObjFileAccess(jobId, fileName);

          if (objExists) {
            console.log(`🎯 Found existing LoD2 model: ${fileName} for job ${jobId}`);

            // Check if this layer already exists
            const existingLayers = layerManagerRef.current.getLayersByType('lod2');
            const alreadyExists = existingLayers.some(layer =>
              layer.metadata?.jobId === jobId && !layer.metadata?.isBMQI
            );

            if (!alreadyExists) {
              // Add the LoD2 model as a layer (including bbox if available)
              const mtlPath = `/outputs/${jobId}/material.mtl`;
              const layerId = addLayer(
                'lod2',
                `LoD2モデル (${job.filename})`,
                {
                  backendGenerated: true,
                  objPath: lod2ObjPath,
                  mtlPath: mtlPath,
                  jobId: jobId,
                  url: `${lod2ObjPath}`,
                  bbox: job.bbox,
                  isBMQI: false
                },
                null,
                {
                  originalLayerName: `LoD2モデル (${job.filename})`,
                  format: 'obj',
                  backendGenerated: true,
                  jobId: jobId,
                  objPath: lod2ObjPath,
                  createdAt: job.completed_at,
                  size: 'Backend processed',
                  verified: true,
                  sourceFile: job.filename,
                  sourceJobId: jobId,
                  bbox: job.bbox,
                  isBMQI: false
                }
              );

              // Log bbox info if available
              if (job.bbox) {
                console.log(`   📍 LoD2 layer has bounding box:`, job.bbox);
              }

              console.log(`✅ Auto-added LoD2 layer: ${layerId} for job ${jobId} (${fileName})`);

              // Auto-enable the layer visibility
              setVisibleLayers(prev => new Set([...prev, layerId]));
              setAvailableLoDs(prev => new Set([...prev, 'lod2']));
              setProjectStage('lod2-generated');
            }

            // Break after finding the first existing file
            break;
          } else {
            console.log(`❌ LoD2 file not found: ${fileName} for job ${jobId}`);
          }
        }

        // Second, check for BMQI LoD2 model
        const lod2BmqiObjPath = `/outputs/${jobId}/${expectedFileNameBMQI}`;
        console.log(`🔍 Checking for LoD2 BMQI file: ${lod2BmqiObjPath}`);

        const bmqiObjExists = await backendApi.testObjFileAccess(jobId, expectedFileNameBMQI);

        if (bmqiObjExists) {
          console.log(`🎯 Found existing LoD2 BMQI model: ${expectedFileNameBMQI} for job ${jobId}`);

          // Check if BMQI layer already exists
          const existingLayers = layerManagerRef.current.getLayersByType('lod2');
          const alreadyExists = existingLayers.some(layer =>
            layer.metadata?.jobId === jobId && layer.metadata?.isBMQI
          );

          if (!alreadyExists) {
            // Add the BMQI LoD2 model as a separate layer
            const mtlPath = `/outputs/${jobId}/material.mtl`;
            const layerId = addLayer(
              'lod2',
              `LoD2モデル (BMQI) (${job.filename})`,
              {
                backendGenerated: true,
                objPath: lod2BmqiObjPath,
                mtlPath: mtlPath,
                jobId: jobId,
                url: `${lod2BmqiObjPath}`,
                bbox: job.bbox,
                isBMQI: true
              },
              null,
              {
                originalLayerName: `LoD2モデル (BMQI) (${job.filename})`,
                format: 'obj',
                backendGenerated: true,
                jobId: jobId,
                objPath: lod2BmqiObjPath,
                createdAt: job.completed_at,
                size: 'Backend processed',
                verified: true,
                sourceFile: job.filename,
                sourceJobId: jobId,
                bbox: job.bbox,
                isBMQI: true
              }
            );

            console.log(`✅ Auto-added LoD2 BMQI layer: ${layerId} for job ${jobId}`);

            // Auto-enable the BMQI layer visibility
            setVisibleLayers(prev => new Set([...prev, layerId]));
          }
        } else {
          console.log(`❌ LoD2 BMQI file not found: ${expectedFileNameBMQI} for job ${jobId}`);
        }
      }
    } catch (error) {
      console.warn('Failed to check for existing LoD2 models:', error);
    }
  };

  // Helper function to add a new layer
  const addLayer = (type, name, data, geojsonId = null, metadata = {}) => {
    const id = generateUniqueId(name, type);
    const layer = createUnifiedLayer(id, type, name, data, geojsonId, metadata);
    
    console.log('🔨 Creating layer:', { id, type, name, data, metadata });
    
    layerManagerRef.current.addLayer(layer);
    triggerLayerUpdate();
    
    // Add to visible layers by default
    setVisibleLayers(prev => new Set([...prev, id]));
    
    console.log(`✅ Added ${type} layer: ${name} (ID: ${id})`);
    console.log('🗂️ Current layer manager state:', layerManagerRef.current.getOrganizedLayers());
    
    return id;
  };

  // Helper function to remove a layer
  const removeLayer = (layerId) => {
    const success = layerManagerRef.current.removeLayer(layerId);
    if (success) {
      triggerLayerUpdate();
      setVisibleLayers(prev => {
        const newSet = new Set(prev);
        newSet.delete(layerId);
        return newSet;
      });
      console.log(`✅ Removed layer: ${layerId}`);
    }
    return success;
  };

  // Get organized layers for LayerPanel
  const getOrganizedLayers = () => {
    const organized = layerManagerRef.current.getOrganizedLayers();

    // Filter out layers that are marked as hidden from panel (e.g., city assets merged with LoD1)
    const filterHiddenLayers = (layers) => {
      return layers.filter(layer => {
        const layerInfo = layerManagerRef.current.getLayer(layer.id);
        return !layerInfo?.metadata?.hiddenFromPanel;
      });
    };

    // Apply filter to all layer types
    organized.geojson = filterHiddenLayers(organized.geojson || []);
    organized.lod1 = filterHiddenLayers(organized.lod1 || []);
    organized.lod2 = filterHiddenLayers(organized.lod2 || []);
    organized.lod3 = filterHiddenLayers(organized.lod3 || []);
    organized.orthophoto = filterHiddenLayers(organized.orthophoto || []);

    // console.log('🗂️ Organized layers from LayerManager:', organized);
    // console.log('🔍 Detailed LoD2 layers:', organized.lod2);

    // For orthophotos, use unified system with legacy fallback only if no unified layers exist
    if (organized.orthophoto.length === 0 && customLayers.length > 0) {
      // Only use legacy customLayers if no unified orthophoto layers exist
      organized.custom = customLayers;
    } else {
      // Use unified orthophoto layers
      organized.custom = organized.orthophoto;
    }
    
    // DEMO MODE: Comment out example GeoJSON, LoD2, and LoD3 layers for clean demo
    /*
    // For the example GeoJSON, use the legacy data if it exists
    if (geojsonData && organized.geojson.length === 0) {
      organized.geojson = [
        { id: 'footprint-example', name: 'Footprint Example', bounds: geojsonData, type: 'geojson' }
      ];
    }
    
    // For LoD2, only show if there are no unified layers AND we have the legacy data
    if (organized.lod2.length === 0) {
      organized.generated = [
        { id: 'Generated-example_lod2', name: 'example_lod2', type: 'lod2' }
      ];
    } else {
      organized.generated = organized.lod2; // Use unified LoD2 layers if they exist
    }
    
    // For LoD3, add legacy layers if no unified layers exist
    if (organized.lod3.length === 0 && lod3Layers.length > 0) {
      organized.lod3 = lod3Layers.map(layer => ({
        id: layer.id,
        name: layer.name,
        type: 'lod3',
        data: layer.data
      }));
    }
    */
    
    // DEMO MODE: Use only unified layers for clean workflow
    organized.generated = organized.lod2; // Use unified LoD2 layers only
    
    // console.log('🏗️ Final organized.generated for LayerPanel:', organized.generated);
    // console.log('📊 All organized layers final:', organized);
    
    return organized;
  };

  // Project workflow state management
  const [projectStage, setProjectStage] = useState('initial'); // 'initial', 'geojson-imported', 'lod1-generated', 'orthophoto-imported', 'pointcloud-imported'
  const [availableLoDs, setAvailableLoDs] = useState(new Set()); // Track which LoDs have been generated
  const [lodData, setLodData] = useState({
    lod1: null,
    lod2: null,
    lod3: null
  });

  // Configuration state
  const [selectedPolygon, setSelectedPolygon] = useState(null); // Currently selected polygon for configuration
  const [configurationData, setConfigurationData] = useState({}); // Store configuration for each polygon

  // Height color toggle state
  const [showHeightColors, setShowHeightColors] = useState(false); // Start with white mode (height colors off)

  // ETA tracking for LoD2 building generation (40-90% progress range)
  const buildingPhaseStartTime = useRef(null);
  const buildingPhaseStartProgress = useRef(null);

  // LoD2 generation progress display
  const [lod2GenerationStatus, setLod2GenerationStatus] = useState(null); // { progress: number, eta: string | null }

  const mapContainerRef = useRef();

  // Utility to reproject coordinates from any CRS to WGS84
  function reprojectGeojsonToWGS84(geojson) {
    // Try to detect CRS from geojson
    let from = 'EPSG:4326';
    if (geojson.crs && geojson.crs.properties && geojson.crs.properties.name) {
      const crsName = geojson.crs.properties.name;
      console.log('Detected CRS name:', crsName);
      
      // Try to extract EPSG code from the name (handles urn:ogc:def:crs:EPSG::30169)
      let match = crsName.match(/EPSG[:/](\d+)/i);
      if (!match) match = crsName.match(/EPSG::(\d+)/i);
      if (match) {
        from = `EPSG:${match[1]}`;
        console.log('Extracted EPSG code:', from);
      } else {
        // fallback: use the full name as proj4 id
        from = crsName;
      }
    }
    
    // Define EPSG:30169 if not already defined
    if (!proj4.defs['EPSG:30169']) {
      // EPSG:30169 - Tokyo / Japan Plane Rectangular CS IX
      // Original definition with Tokyo datum, transforming to WGS84
      proj4.defs('EPSG:30169', '+proj=tmerc +lat_0=36 +lon_0=139.8333333333333 +k=0.9999 +x_0=0 +y_0=0 +ellps=bessel +towgs84=-146.414,507.337,680.507,0,0,0,0 +units=m +no_defs');
      console.log('Defined EPSG:30169 projection with Tokyo datum and WGS84 transformation parameters');
    }
    
    // Also define JGD2000 version
    if (!proj4.defs['EPSG:6669']) {
      // JGD2000 / Japan Plane Rectangular CS IX
      proj4.defs('EPSG:6669', '+proj=tmerc +lat_0=36 +lon_0=139.8333333333333 +k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs');
      console.log('Defined EPSG:6669 (JGD2000 / Japan Plane Rectangular CS IX)');
    }
    
    // Define EPSG:6668 - JGD2000 / Japan Plane Rectangular CS VIII
    if (!proj4.defs['EPSG:6668']) {
      // JGD2000 / Japan Plane Rectangular CS VIII
      proj4.defs('EPSG:6668', '+proj=tmerc +lat_0=36 +lon_0=138.5 +k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs');
      console.log('Defined EPSG:6668 (JGD2000 / Japan Plane Rectangular CS VIII)');
    }
    
    const to = 'EPSG:4326';
    
    function reprojectCoords(coords) {
      // Safety check for undefined/null coordinates
      if (!coords || !Array.isArray(coords) || coords.length === 0) {
        console.warn('Invalid coordinates provided to reprojectCoords:', coords);
        return coords || [];
      }
      
      if (typeof coords[0] === 'number') {
        if (from === to) {
          return coords;
        }
        try {
          // Add timeout protection for coordinate transformation
          const transformWithTimeout = (from, to, coords, timeout = 5000) => {
            return new Promise((resolve, reject) => {
              const timer = setTimeout(() => {
                reject(new Error('Coordinate transformation timeout'));
              }, timeout);
              
              try {
                const result = proj4(from, to, coords);
                clearTimeout(timer);
                resolve(result);
              } catch (error) {
                clearTimeout(timer);
                reject(error);
              }
            });
          };
          
          // For synchronous operation, just use regular proj4 but with validation
          const transformed = proj4(from, to, coords);
          
          // Strict coordinate validation for WGS84
          if (transformed && transformed.length >= 2) {
            let [lon, lat] = transformed;
            
            // Validate that we got actual numbers, not NaN or infinity
            if (!isFinite(lon) || !isFinite(lat)) {
              console.warn('Invalid transformation result (non-finite):', transformed, 'from:', coords);
              return coords; // Return original if transformation produced invalid numbers
            }
            
            // Clamp longitude to valid range [-180, 180]
            if (lon < -180) lon = -180;
            if (lon > 180) lon = 180;
            
            // Clamp latitude to valid range [-90, 90]
            if (lat < -90) lat = -90;
            if (lat > 90) lat = 90;
            
            // Additional validation for Japan region
            if (lon >= 120 && lon <= 155 && lat >= 20 && lat <= 50) {
              return [lon, lat];
            } else {
              console.warn('Coordinates out of Japan range, but within valid WGS84 bounds:', [lon, lat]);
              // Try alternative JGD2000 transformations
              if (from === 'EPSG:30169') {
                try {
                  // Try EPSG:6669 (JGD2000 CS IX) first
                  const jgd2000Transform = proj4('EPSG:6669', 'EPSG:4326', coords);
                  if (jgd2000Transform && jgd2000Transform.length >= 2 && 
                      isFinite(jgd2000Transform[0]) && isFinite(jgd2000Transform[1])) {
                    let [jLon, jLat] = jgd2000Transform;
                    
                    // Clamp JGD2000 results as well
                    if (jLon < -180) jLon = -180;
                    if (jLon > 180) jLon = 180;
                    if (jLat < -90) jLat = -90;
                    if (jLat > 90) jLat = 90;
                    
                    if (jLon >= 120 && jLon <= 155 && jLat >= 20 && jLat <= 50) {
                      console.log(`✅ JGD2000 CS IX transformation successful: ${coords} -> [${jLon}, ${jLat}]`);
                      return [jLon, jLat];
                    }
                  }
                  
                  // Try EPSG:6668 (JGD2000 CS VIII) as second alternative
                  const jgd2000Transform2 = proj4('EPSG:6668', 'EPSG:4326', coords);
                  if (jgd2000Transform2 && jgd2000Transform2.length >= 2 &&
                      isFinite(jgd2000Transform2[0]) && isFinite(jgd2000Transform2[1])) {
                    let [jLon2, jLat2] = jgd2000Transform2;
                    
                    // Clamp JGD2000 results as well
                    if (jLon2 < -180) jLon2 = -180;
                    if (jLon2 > 180) jLon2 = 180;
                    if (jLat2 < -90) jLat2 = -90;
                    if (jLat2 > 90) jLat2 = 90;
                    
                    if (jLon2 >= 120 && jLon2 <= 155 && jLat2 >= 20 && jLat2 <= 50) {
                      console.log(`✅ JGD2000 CS VIII transformation successful: ${coords} -> [${jLon2}, ${jLat2}]`);
                      return [jLon2, jLat2];
                    }
                  }
                } catch (error) {
                  console.warn('JGD2000 alternative transformations failed:', error);
                }
              }
              // Return clamped coordinates even if out of Japan range
              return [lon, lat];
            }
          } else {
            console.error('Invalid transformation result:', transformed);
            return coords; // Return original if transformation failed
          }
        } catch (error) {
          console.error('Coordinate transformation error:', error);
          return coords;
        }
      } else {
        return coords.map(reprojectCoords);
      }
    }
    
    const reprojectedGeojson = {
      ...geojson,
      crs: undefined, // Remove CRS property for Mapbox compatibility
      features: geojson.features.map((f, index) => ({
        ...f,
        id: f.id || f.properties?.id || `feature-${index}`, // Ensure each feature has an ID
        properties: {
          ...f.properties,
          id: f.properties?.id || f.id || `feature-${index}` // Also set ID in properties
        },
        geometry: {
          ...f.geometry,
          coordinates: f.geometry && f.geometry.coordinates ? reprojectCoords(f.geometry.coordinates) : []
        }
      }))
    };
    
    console.log('=== COORDINATE TRANSFORMATION DEBUG ===');
    console.log('Source CRS:', from);
    console.log('Target CRS:', to);
    console.log('Original first feature coords (first 3 points):');
    if (geojson.features.length > 0 && geojson.features[0].geometry.coordinates[0]) {
      const firstRing = geojson.features[0].geometry.coordinates[0];
      for (let i = 0; i < Math.min(3, firstRing.length); i++) {
        console.log(`  Point ${i}:`, firstRing[i]);
      }
    }
    console.log('Reprojected first feature coords (first 3 points):');
    if (reprojectedGeojson.features.length > 0 && reprojectedGeojson.features[0].geometry.coordinates[0]) {
      const firstRing = reprojectedGeojson.features[0].geometry.coordinates[0];
      for (let i = 0; i < Math.min(3, firstRing.length); i++) {
        console.log(`  Point ${i}:`, firstRing[i]);
      }
    }
    
    // Calculate bounds for validation with safety checks
    let minLon = Infinity, maxLon = -Infinity, minLat = Infinity, maxLat = -Infinity;
    reprojectedGeojson.features.forEach(feature => {
      if (feature.geometry && feature.geometry.coordinates && feature.geometry.coordinates[0]) {
        feature.geometry.coordinates[0].forEach(coord => {
          if (Array.isArray(coord) && coord.length >= 2) {
            const [lon, lat] = coord;
            // Only process valid coordinates
            if (typeof lon === 'number' && typeof lat === 'number' && 
                !isNaN(lon) && !isNaN(lat) && 
                lon >= -180 && lon <= 180 && 
                lat >= -90 && lat <= 90) {
              minLon = Math.min(minLon, lon);
              maxLon = Math.max(maxLon, lon);
              minLat = Math.min(minLat, lat);
              maxLat = Math.max(maxLat, lat);
            }
          }
        });
      }
    });
    console.log('Calculated bounds:', { minLon, maxLon, minLat, maxLat });
    console.log('Expected Tokyo bounds: lon 139.6-139.8, lat 35.6-35.8');
    console.log('========================================');
    
    return reprojectedGeojson;
  }

  // DEMO MODE: Comment out example data initialization for clean demo workflow
  /*
  useEffect(() => {
    console.log('🚀 Initializing example data layers (original method)...');
    
    // Load example GeoJSON using original method only
    fetch('/src/assets/test_data/01/footprint_1.geojson')
      .then(res => res.json())
      .then(rawGeojson => {
        const reprojectedGeojson = reprojectGeojsonToWGS84(rawGeojson);
        setGeojsonData(reprojectedGeojson);
        
        console.log('✅ Loaded example GeoJSON data (original method)');
        
        // Keep legacy LoD3 layers for backward compatibility - use correct paths
        const lod3Example = {
          id: 'lod3-example-route1',
          name: 'LoD3 Route1 Example',
          data: {
            objPath: '/src/assets/test_data/01/lod3/route1/results_route1_lod3_350m.obj',
            mtlPath: '/src/assets/test_data/01/lod3/route1/material.mtl'
          },
          metadata: {
            description: 'LoD3 model with materials',
            source: 'test_data/01/lod3/route1'
          }
        };
        
        setLod3Layers([lod3Example]);
        
        console.log('✅ Added LoD3 example layer (original method)');
      })
      .catch(err => console.error('❌ Failed to load example GeoJSON:', err));
  }, []);
  */

  // Clean up uploads and outputs directories on page load/refresh
  // useEffect(() => {
  //   const performCleanup = async () => {
  //     try {
  //       await backendApi.cleanup();
  //     } catch (error) {
  //       // Silently ignore cleanup errors (backend might not be ready yet)
  //     }
  //   };
  //   performCleanup();
  // }, []);

  // Enhanced zoom function using original logic + unified layer management
  const zoomToLayer = useCallback((layerId) => {
    console.log(`🎯 Zooming to layer: ${layerId}`);
    
    // Handle legacy footprint example (original logic)
    if (layerId === 'footprint-example' && mapContainerRef.current) {
      mapContainerRef.current.zoomToGeojson();
      return;
    }
    
    // Handle legacy LoD2 example - now use OBJ coordinates
    if (layerId === 'Generated-example_lod2' && mapContainerRef.current) {
      console.log('🎯 Zooming to LoD2 example using OBJ coordinates');
      if (mapContainerRef.current.zoomToObjModel) {
        mapContainerRef.current.zoomToObjModel('/src/assets/test_data/01/result_lod2.obj', layerId);
      } else {
        // Fallback to GeoJSON zoom if OBJ zoom not available
        mapContainerRef.current.zoomToGeojson();
      }
      return;
    }
    
    // Handle LoD3 layers - now use OBJ coordinates
    const lod3Layer = lod3Layers.find(layer => layer.id === layerId);
    if (lod3Layer && mapContainerRef.current) {
      console.log('🎯 Zooming to LoD3 layer using OBJ coordinates:', layerId);
      if (mapContainerRef.current.zoomToObjModel && lod3Layer.data?.objPath) {
        mapContainerRef.current.zoomToObjModel(lod3Layer.data.objPath, layerId);
      } else {
        console.warn('⚠️ LoD3 layer missing objPath, falling back to GeoJSON zoom');
        mapContainerRef.current.zoomToGeojson();
      }
      return;
    }

    // Handle imported GeoJSON layers (original logic)
    const geojsonLayer = importedGeojsonLayers.find(layer => layer.id === layerId);
    if (geojsonLayer && mapContainerRef.current) {
      const currentGeojsonData = geojsonData;
      const currentShowGeojson = showGeojson;
      setGeojsonData(geojsonLayer.data);
      setShowGeojson(true);
      setTimeout(() => {
        mapContainerRef.current.zoomToGeojson();
        setTimeout(() => {
          setGeojsonData(currentGeojsonData);
          setShowGeojson(currentShowGeojson);
        }, 1000);
      }, 100);
      return;
    }
    
    // Get layer from unified manager for new layers
    const layer = layerManagerRef.current.getLayer(layerId);
    
    if (layer && mapContainerRef.current) {
      switch (layer.type) {
        case 'geojson':
          // For GeoJSON layers, zoom to their bounds
          if (layer.data && mapContainerRef.current.zoomToGeojson) {
            const currentGeojsonData = geojsonData;
            const currentShowGeojson = showGeojson;
            setGeojsonData(layer.data);
            setShowGeojson(true);
            setTimeout(() => {
              mapContainerRef.current.zoomToGeojson();
              // Restore original data after zoom
              setTimeout(() => {
                setGeojsonData(currentGeojsonData);
                setShowGeojson(currentShowGeojson);
              }, 1000);
            }, 100);
          }
          break;
          
        case 'lod1':
          console.log('🎯 Zooming to LoD1 layer:', layerId);
          // For LoD1, prefer using OBJ coordinate-based zoom if available
          if (mapContainerRef.current.zoomToObjModel && layer.metadata?.objPath) {
            console.log('   ├─ Using OBJ coordinate-based zoom');
            mapContainerRef.current.zoomToObjModel(layer.metadata.objPath, layerId);
          } else if (layer.geojsonId) {
            // Fallback: zoom to the source GeoJSON if available
            console.log('   ├─ Falling back to source GeoJSON zoom');
            const sourceLayer = layerManagerRef.current.getLayer(layer.geojsonId);

            if (sourceLayer && sourceLayer.data) {
              console.log(`   └─ Zooming to source GeoJSON: ${sourceLayer.name}`);
              const currentGeojsonData = geojsonData;
              const currentShowGeojson = showGeojson;
              setGeojsonData(sourceLayer.data);
              setShowGeojson(true);
              setTimeout(() => {
                mapContainerRef.current.zoomToGeojson();
                setTimeout(() => {
                  setGeojsonData(currentGeojsonData);
                  setShowGeojson(currentShowGeojson);
                }, 1000);
              }, 100);
            } else {
              console.warn('⚠️ Source GeoJSON layer not found for LoD1');
            }
          } else {
            console.warn('⚠️ LoD1 layer missing objPath and geojsonId, cannot zoom');
          }
          break;
          
        case 'lod2':
          console.log('🎯 Zooming to LoD2 layer using OBJ coordinates:', layerId);
          // For LoD2, use OBJ coordinate-based zoom
          if (mapContainerRef.current.zoomToObjModel && layer.metadata?.objPath) {
            mapContainerRef.current.zoomToObjModel(layer.metadata.objPath, layerId);
          } else {
            console.warn('⚠️ LoD2 layer missing objPath metadata, falling back to data zoom');
            // Fallback to zooming to layer data bounds
            if (layer.data && mapContainerRef.current.zoomToGeojson) {
              const currentGeojsonData = geojsonData;
              const currentShowGeojson = showGeojson;
              setGeojsonData(layer.data);
              setShowGeojson(true);
              setTimeout(() => {
                mapContainerRef.current.zoomToGeojson();
                setTimeout(() => {
                  setGeojsonData(currentGeojsonData);
                  setShowGeojson(currentShowGeojson);
                }, 1000);
              }, 100);
            }
          }
          break;
          
        case 'lod3':
          console.log('🎯 Zooming to LoD3 layer using OBJ coordinates:', layerId);
          // For LoD3, use OBJ coordinate-based zoom
          if (mapContainerRef.current.zoomToObjModel && layer.metadata?.objPath) {
            mapContainerRef.current.zoomToObjModel(layer.metadata.objPath, layerId);
          } else {
            console.warn('⚠️ LoD3 layer missing objPath metadata, falling back to data zoom');
            // Fallback to zooming to layer data bounds
            if (layer.data && mapContainerRef.current.zoomToGeojson) {
              const currentGeojsonData = geojsonData;
              const currentShowGeojson = showGeojson;
              setGeojsonData(layer.data);
              setShowGeojson(true);
              setTimeout(() => {
                mapContainerRef.current.zoomToGeojson();
                setTimeout(() => {
                  setGeojsonData(currentGeojsonData);
                  setShowGeojson(currentShowGeojson);
                }, 1000);
              }, 100);
            }
          }
          break;
          
        case 'orthophoto':
          console.log('🎯 Zooming to orthophoto layer:', layerId);
          if (mapContainerRef.current.zoomToOrthophoto) {
            mapContainerRef.current.zoomToOrthophoto(layerId);
          }
          break;
      }
    }
  }, [lod3Layers, importedGeojsonLayers, layerManagerRef, mapContainerRef, geojsonData, showGeojson]);

  // Function to zoom to an orthophoto layer
  const zoomToOrthophoto = useCallback((layerId) => {
    if (mapContainerRef.current) {
      mapContainerRef.current.zoomToOrthophoto(layerId);
    }
  }, [mapContainerRef]);

  // Event handlers
  const handleToolChange = useCallback((toolId) => {
    if (toolId === 'layers') {
      setShowLayerPanel(prev => !prev);
      // Only set activeTool to 'layers' when opening the panel
      if (!showLayerPanel) {
        setActiveTool('layers');
      } else {
        setActiveTool(null);
      }
    } else {
      setActiveTool(toolId);

      // Handle tool-specific actions
      switch (toolId) {
        case 'zoom-in':
          mapInstance?.zoomIn();
          break;
        case 'zoom-out':
          mapInstance?.zoomOut();
          break;
        case 'search':
          // Implement search functionality
          console.log('Search activated');
          break;
        case 'grid':
          // Toggle grid/measurement tools
          console.log('Grid tools activated');
          break;
        default:
          break;
      }
    }
  }, [showLayerPanel, mapInstance]);

  const handleMapLoad = useCallback((map) => {
    setMapInstance(map);
  }, []);

  const handleBuildingClick = useCallback((feature) => {
    // Background building clicks no longer activate configuration mode
    setSelectedBuilding(feature);
    // Configuration mode is now only activated by clicking the LoD1 button
  }, []);

  // Configuration handling functions
  const handlePolygonClick = useCallback((polygon) => {
    // Only handle polygon clicks in configuration mode
    if (mapState === 'configuration') {
      setSelectedPolygon(polygon);
    }
    // In other modes, do nothing - don't restore buildings or change anything
  }, [mapState]);

  const handleConfigurationUpdate = useCallback((polygonId, config) => {
    setConfigurationData(prev => ({
      ...prev,
      [polygonId]: config
    }));

    // Update the polygon properties in geojsonData using the utility function
    if (geojsonData) {
      const updatedGeoJson = updatePolygonHeight(geojsonData, polygonId, config.height);
      setGeojsonData(updatedGeoJson);

      // Log height statistics for debugging
      const stats = getHeightStatistics(updatedGeoJson);
      console.log('Updated GeoJSON height statistics:', stats);
    }
  }, [geojsonData]);

  const handleGenerateLoD1 = useCallback(() => {
    if (geojsonData && currentGeojsonFile) {
      console.log('🔨 Generating LoD1 model...');
      
      // Create LoD1 data with the configured heights
      const lodData = {
        type: 'FeatureCollection',
        features: geojsonData.features.map(feature => ({
          ...feature,
          properties: {
            ...feature.properties,
            lodLevel: 'lod1',
            configured: true
          }
        }))
      };

      // Find the source GeoJSON layer to establish relationship
      const sourceGeojsonLayer = layerManagerRef.current.getAllLayers().find(
        layer => layer.type === 'geojson' && layer.metadata?.originalName === currentGeojsonFile.name
      );
      const sourceGeojsonId = sourceGeojsonLayer?.id || null;
      
      // Generate name based on source GeoJSON
      const lod1Name = generateLoDModelName(currentGeojsonFile.name, { height: 'configured' });
      
      // Add to unified layer management system
      const lod1Id = addLayer(
        'lod1',
        lod1Name,
        lodData,
        sourceGeojsonId, // Link to source GeoJSON
        {
          sourceGeojsonFile: currentGeojsonFile.name,
          generatedFrom: geojsonData,
          lodLevel: 'lod1',
          configurationData: configurationData // Store configuration data
        }
      );

      // Keep legacy state for backward compatibility
      // Get the unified layer to use its controlId
      const unifiedLayer = layerManagerRef.current.getLayer(lod1Id);
      console.log(`🔍 DEBUG: LoD1 unified layer:`, unifiedLayer);
      console.log(`🔍 DEBUG: Unified controlId: ${unifiedLayer?.controlId}`);

      const lod1Layer = {
        id: lod1Id,
        name: lod1Name,
        type: 'lod1',
        data: lodData,
        controlId: unifiedLayer?.controlId || lod1Id, // Use unified controlId at top level
        metadata: {
          sourceGeojsonFile: currentGeojsonFile.name,
          generatedFrom: geojsonData,
          lodLevel: 'lod1'
        }
      };

      console.log(`📦 Legacy LoD1 layer created:`, {
        id: lod1Layer.id,
        controlId: lod1Layer.controlId,
        hasControlId: !!lod1Layer.controlId
      });
      setLod1Layers(prev => [...prev, lod1Layer]);
      
      // Add to available LoDs and update project stage
      setAvailableLoDs(prev => new Set([...prev, 'lod1']));
      setLodData(prev => ({
        ...prev,
        lod1: lodData
      }));

      setProjectStage('lod1-generated');
      // Keep configuration mode active so user can continue configuring other polygons
      // setMapState('normal'); // Removed: stay in configuration mode
      setSelectedPolygon(null); // Clear selection but stay in configuration mode

      console.log(`✅ LoD1 model generated successfully: ${lod1Id}`);
    }
  }, [geojsonData, currentGeojsonFile, layerManagerRef, addLayer, configurationData]);

  const handleCloseConfiguration = useCallback(() => {
    // Toggle GeoJSON layer off, which will automatically exit configuration mode
    const geojsonLayers = layerManagerRef.current.getLayersByType('geojson');
    if (geojsonLayers.length > 0) {
      const geojsonLayerId = geojsonLayers[0].id;
      console.log(`🔇 Closing configuration - toggling GeoJSON layer OFF: ${geojsonLayerId}`);
      setVisibleLayers(prev => {
        const newSet = new Set(prev);
        newSet.delete(geojsonLayerId);
        return newSet;
      });
    }
    setMapState('normal');
    setSelectedPolygon(null);
    setShowGeojson(false);
  }, [layerManagerRef]);

  const handleLoDSelect = useCallback((lodLevel) => {
    console.log('LoD selected:', lodLevel);

    if (lodLevel === 'lod1' && projectStage === 'geojson-imported') {
      // Toggle GeoJSON layer on to enter configuration mode
      // Find the GeoJSON layer and toggle it
      const geojsonLayers = layerManagerRef.current.getLayersByType('geojson');
      if (geojsonLayers.length > 0) {
        const geojsonLayerId = geojsonLayers[0].id;
        console.log(`🎯 LoD1 selected - toggling GeoJSON layer ON: ${geojsonLayerId}`);
        // Add to visible layers and enter configuration mode
        setVisibleLayers(prev => new Set([...prev, geojsonLayerId]));
        setMapState('configuration');
        setShowGeojson(true);
        console.log('Entering LoD1 configuration mode - toggle GeoJSON OFF to exit');
      }
    } else if (lodLevel === 'lod2' && availableLoDs.has('lod1')) {
      // Open import modal for orthophoto
      setShowImportModel(true);
      console.log('Ready for LoD2 - import orthophoto data');
    } else if (lodLevel === 'lod3' && availableLoDs.has('lod2')) {
      // Open import modal for point cloud
      setShowImportModel(true);
      console.log('Ready for LoD3 - import point cloud data');
    }
  }, [projectStage, availableLoDs, layerManagerRef]);

  const handleLayerToggle = useCallback((layerId, isVisible) => {
    console.log(`🎛️ Toggle layer: ${layerId} -> ${isVisible ? 'visible' : 'hidden'}`);

    // Validate layerId
    if (!layerId || typeof layerId !== 'string' || layerId.trim() === '') {
      console.error('❌ Invalid layer ID provided to handleLayerToggle:', layerId);
      return;
    }

    // Find all layers that share the same controlId (for merged layers like LoD1 + City Assets)
    const toggledLayer = layerManagerRef.current.getLayer(layerId);
    const controlId = toggledLayer?.controlId;
    const layersToToggle = [layerId];

    if (controlId) {
      // Find all other layers with the same controlId
      const allLayers = layerManagerRef.current.getAllLayers();
      allLayers.forEach(layer => {
        if (layer.controlId === controlId && layer.id !== layerId) {
          layersToToggle.push(layer.id);
          console.log(`   ├─ Also toggling merged layer: ${layer.id}`);
        }
      });
    }

    // Check if the toggled layer is a GeoJSON layer using unified layer manager first
    const toggledLayerInfo = layerManagerRef.current.getLayer(layerId);
    const isGeoJsonLayer = toggledLayerInfo?.type === 'geojson';

    // Update visible layers state for all layers in the group
    const newVisibleLayers = new Set(visibleLayers);
    layersToToggle.forEach(id => {
      if (isVisible) {
        newVisibleLayers.add(id);
        // Only set layout property for layers that exist in the map
        if (mapInstance && mapInstance.getLayer && mapInstance.getLayer(id)) {
          mapInstance.setLayoutProperty(id, 'visibility', 'visible');
        }
      } else {
        newVisibleLayers.delete(id);
        if (mapInstance && mapInstance.getLayer && mapInstance.getLayer(id)) {
          mapInstance.setLayoutProperty(id, 'visibility', 'none');
        }
      }
    });

    // Handle legacy layers first (original logic)
    if (layerId === 'footprint-example') {
      setShowGeojson(isVisible);
      setVisibleLayers(newVisibleLayers);
      return;
    }

    if (layerId === 'Generated-example_lod2') {
      // For LoD2 example, we'll handle it via the showObjModel state in MapContainer
      console.log(`🏗️ LoD2 example layer ${layerId} ${isVisible ? 'shown' : 'hidden'}`);
      setVisibleLayers(newVisibleLayers);
      return;
    }

    // Handle GeoJSON layer toggle - enter/exit configuration mode
    if (isGeoJsonLayer) {
      if (isVisible) {
        console.log(`📍 GeoJSON layer toggled ON - entering configuration mode`);

        // Hide all LoD1 layers when entering configuration mode FIRST
        console.log(`🔇 Hiding all LoD1 layers before entering configuration mode`);
        const lod1Layers = layerManagerRef.current.getLayersByType('lod1');
        console.log(`   ├─ Found ${lod1Layers.length} LoD1 layers to hide`);
        lod1Layers.forEach(lod1Layer => {
          newVisibleLayers.delete(lod1Layer.id);
          console.log(`   ├─ Removed from visibleLayers: ${lod1Layer.id}`);
        });

        // Set the geojsonData from the unified layer manager
        if (toggledLayerInfo && toggledLayerInfo.data) {
          console.log(`   ├─ Setting GeoJSON data from layer: ${layerId}`);
          setGeojsonData(toggledLayerInfo.data);
          setShowGeojson(true);

          // Set currentGeojsonFile for regeneration to work
          if (toggledLayerInfo.metadata && toggledLayerInfo.metadata.originalName) {
            setCurrentGeojsonFile({ name: toggledLayerInfo.metadata.originalName });
            console.log(`   ├─ Set current GeoJSON file: ${toggledLayerInfo.metadata.originalName}`);
          }
        } else {
          console.warn(`   ⚠️ No GeoJSON data found for layer: ${layerId}`);
        }

        // Debug: Check final state
        console.log(`📍 GeoJSON layer ID: ${layerId}`);
        console.log(`📍 GeoJSON in newVisibleLayers: ${newVisibleLayers.has(layerId)}`);
        console.log(`📍 Final newVisibleLayers (${newVisibleLayers.size} items):`, Array.from(newVisibleLayers));
        console.log(`📍 About to set states - showGeojson: true, mapState: configuration`);

        // Update visible layers FIRST so MapContainer gets the updated visibleLayers
        setVisibleLayers(newVisibleLayers);

        // Then set mapState which will trigger MapContainer to re-render with correct states
        setMapState('configuration');
      } else {
        console.log(`📍 GeoJSON layer toggled OFF - exiting configuration mode`);
        setMapState('normal');
        setShowGeojson(false);
        setSelectedPolygon(null); // Clear any selected polygon
        // Hide the Mapbox layers directly
        if (mapInstance && mapInstance.getLayer) {
          const geojsonLayerIds = ['geojson-mask-extrusion', 'geojson-mask-fill', 'geojson-mask-stroke'];
          geojsonLayerIds.forEach(mapLayerId => {
            if (mapInstance.getLayer(mapLayerId)) {
              mapInstance.setLayoutProperty(mapLayerId, 'visibility', 'none');
              console.log(`   ├─ Hidden: ${mapLayerId}`);
            }
          });
        }
        // For toggle OFF, also update visible layers
        setVisibleLayers(newVisibleLayers);
      }
      return;
    }
    
    // Handle LoD3 layers (original logic)
    const lod3Layer = lod3Layers.find(layer => layer.id === layerId);
    if (lod3Layer) {
      console.log(`🏗️ LoD3 layer ${layerId} ${isVisible ? 'shown' : 'hidden'}`);
      // The visibility is handled by the visibleLayers state in MapContainer
      setVisibleLayers(newVisibleLayers);
      return;
    }
    
    // Get layer from unified manager for new layers
    const layer = layerManagerRef.current.getLayer(layerId);
    if (layer) {
      // Update unified layer manager
      layerManagerRef.current.setLayerVisibility(layerId, isVisible);
      triggerLayerUpdate();
      
      switch (layer.type) {
        case 'geojson':
          // Handle GeoJSON visibility
          if (isVisible) {
            setGeojsonData(layer.data);
            setShowGeojson(true);
          } else {
            // Hide the Mapbox layers directly instead of using showGeojson
            console.log('🔇 Hiding GeoJSON layers (unified layer)...');
            if (mapInstance && mapInstance.getLayer) {
              const geojsonLayerIds = ['geojson-mask-extrusion', 'geojson-mask-fill', 'geojson-mask-stroke'];
              geojsonLayerIds.forEach(mapLayerId => {
                if (mapInstance.getLayer(mapLayerId)) {
                  mapInstance.setLayoutProperty(mapLayerId, 'visibility', 'none');
                  console.log(`   ├─ Hidden: ${mapLayerId}`);
                } else {
                  console.log(`   ├─ Layer not found: ${mapLayerId}`);
                }
              });
            } else {
              console.warn('   ⚠️ mapInstance not available for hiding GeoJSON layers');
            }
          }
          break;
          
        case 'lod1':
        case 'lod2': 
        case 'lod3':
          // Handle LoD model visibility
          console.log(`🏗️ LoD ${layer.type} layer ${layerId} ${isVisible ? 'shown' : 'hidden'}`);
          break;
          
        case 'orthophoto':
          console.log(`🖼️ Orthophoto layer ${layerId} ${isVisible ? 'shown' : 'hidden'}`);
          break;
      }
    }

    // Update visible layers for any remaining cases
    setVisibleLayers(newVisibleLayers);
  }, [visibleLayers, mapInstance, importedGeojsonLayers, lod3Layers, layerManagerRef, triggerLayerUpdate]);

  const handleImportClick = useCallback(() => {
    setShowImportModel(true);
  }, []);

  const handleImportComplete = async (importData) => {
    console.log('🚀 handleImportComplete called', importData);
    console.log('📦 Import data type:', importData.type);
    console.log('📦 Import data keys:', Object.keys(importData));
    
    // Handle backend-generated LoD2 model
    if (importData.type === 'lod2-backend-generated') {
      console.log('⚙️ Processing Backend LoD2 model:', importData);
      console.log('🔍 Backend LoD2 Details:', {
        jobId: importData.backend.jobId,
        objPath: importData.backend.objPath,
        fullUrl: `${importData.backend.objPath}`,
        verified: importData.backend.verified
      });
      
      console.log('🔧 About to call addLayer with type="lod2"...');
      
      // Create layer for the backend-generated LoD2 model
      const layerId = addLayer(
        'lod2',
        `Backend LoD2 Model (${importData.backend.jobId})`,
        {
          backendGenerated: true,
          objPath: importData.backend.objPath,
          jobId: importData.backend.jobId,
          url: `${importData.backend.objPath}`
        },
        null,
        {
          originalLayerName: `Backend LoD2 Model (${importData.backend.jobId})`,
          format: 'obj',
          backendGenerated: true,
          jobId: importData.backend.jobId,
          objPath: importData.backend.objPath,
          createdAt: new Date().toISOString(),
          size: 'Backend processed',
          config: importData.config,
          verified: importData.backend.verified
        }
      );
      
      console.log('🏗️ Created LoD2 layer:', {
        layerId: layerId,
        jobId: importData.backend.jobId,
        objUrl: `${importData.backend.objPath}`
      });
      
      // Auto-enable the backend LoD2 layer visibility
      setVisibleLayers(prev => new Set([...prev, layerId]));
      console.log(`🎭 Auto-enabled backend LoD2 layer visibility: ${layerId}`);
      
      setAvailableLoDs(prev => new Set([...prev, 'lod2']));
      setProjectStage('lod2-generated');
      setShowImportModel(false);
      console.log(`✅ Backend LoD2 model added: ${layerId}`);
      return;
    }
    
    // Handle LoD generation
    if (importData.type === 'lod-generation') {
      console.log('⚙️ Configuring LoD:', importData.lodType, 'with config:', importData.config);
      
      // For LoD1, enter configuration mode instead of directly generating
      if (importData.lodType === 'lod1') {
        setShowImportModel(false);
        setMapState('configuration');
        setShowGeojson(true); // Ensure GeoJSON is visible in configuration mode
        console.log('Entering LoD1 configuration mode - click on GeoJSON polygons to configure height');
        return;
      }
      
      // For LoD2 and LoD3, use the traditional generation approach
      setAvailableLoDs(prev => new Set([...prev, importData.lodType]));
      
      // Store LoD data
      setLodData(prev => ({
        ...prev,
        [importData.lodType]: {
          config: importData.config,
          sourceType: importData.sourceType,
          generated: true
        }
      }));
      
      // Update project stage
      if (importData.lodType === 'lod2') {
        setProjectStage('lod2-generated');
      } else if (importData.lodType === 'lod3') {
        setProjectStage('lod3-generated');
      }
      
      setShowImportModel(false);
      return;
    }
    
    // Handle demo TIFF import
    if (importData.type === 'demo-tiff') {
      console.log('📷 Processing Demo TIFF...');
      
      // First, load demo GeoJSON data for building hiding functionality
      try {
        const response = await fetch('/src/assets/test_data/01/footprint_1.geojson');
        const rawGeojson = await response.json();
        const reprojectedGeojson = reprojectGeojsonToWGS84(rawGeojson);
        setGeojsonData(reprojectedGeojson);
        console.log('✅ Loaded demo GeoJSON data for building hiding');
      } catch (error) {
        console.warn('⚠️ Failed to load demo GeoJSON:', error);
      }
      
      // Load demo orthophoto data
      const demoTiffPath = '/src/assets/result_demo/tiff/route1.tif';
      
      // Create demo bounds (approximate Tokyo area)
      const demoBounds = [
        [139.6, 35.6], // Southwest
        [139.8, 35.6], // Southeast  
        [139.8, 35.8], // Northeast
        [139.6, 35.8]  // Northwest
      ];
      
      // Add to unified layer management system
      const layerId = addLayer(
        'orthophoto',
        importData.layerName,
        {
          demoMode: true,
          demoPath: demoTiffPath,
          coordinates: demoBounds,
          previewUrl: '/src/assets/result_demo/tiff/route1.tif' // Use the existing preview
        },
        null,
        {
          originalLayerName: importData.layerName,
          format: 'demo-tiff',
          bounds: demoBounds,
          demoMode: true,
          createdAt: new Date().toISOString(),
          size: 'Demo file'
        }
      );
      
      // Add to map
      // if (mapInstance && !mapInstance.getSource(layerId)) {
      //   mapInstance.addSource(layerId, {
      //     type: 'image',
      //     url: '/src/assets/result_demo/tiff/route1.tif',
      //     coordinates: demoBounds,
      //   });
      //   mapInstance.addLayer({
      //     id: layerId,
      //     type: 'raster',
      //     source: layerId,
      //     paint: {},
      //   }, '3d-buildings');
        
      //   // Register the layer with the layer control system
      //   if (mapContainerRef.current?.registerExternalLayer) {
      //     mapContainerRef.current.registerExternalLayer(layerId, layerId);
      //   }
      // }
      
      console.log(`✅ Added Demo TIFF layer: ${layerId}`);
      
      // DEMO MODE: Auto-generate LoD2 model after demo TIFF import
      setTimeout(() => {
        console.log('🚀 Demo: Auto-generating LoD2 model...');
        const demoLod2Path = '/src/assets/result_demo/result/r1l2/results_route1_lod2.obj';
        
        const layerId = addLayer(
          'lod2',
          'Demo LoD2 Model',
          { 
            demoMode: true,
            demoPath: demoLod2Path,
            objPath: demoLod2Path,
            url: demoLod2Path 
          },
          null,
          {
            originalLayerName: 'Demo LoD2 Model',
            format: 'obj',
            demoMode: true,
            autoGenerated: true,
            createdAt: new Date().toISOString(),
            size: 'Demo file'
          }
        );
        
        // ✅ FIX: Auto-enable the demo LoD2 layer visibility
        setVisibleLayers(prev => new Set([...prev, layerId]));
        console.log(`🎭 Auto-enabled demo LoD2 layer visibility: ${layerId}`);
        
        setAvailableLoDs(prev => new Set([...prev, 'lod2']));
        setProjectStage('orthophoto-imported');
        console.log(`✅ Demo LoD2 model added: ${layerId}`);
      }, 1500);
      
      setShowImportModel(false);
      return;
    }
    
    // Handle LoD3 combined data import (pointcloud folder + streetview folder)
    if (importData.type === 'lod3-data') {
      console.log('🏗️ Processing LoD3 combined data (folders)...');
      console.log('📦 Folders:', {
        pointcloud: `${importData.files?.pointcloud?.folderName} (${importData.files?.pointcloud?.fileCount} files)`,
        streetview: `${importData.files?.streetview?.folderName} (${importData.files?.streetview?.fileCount} files)`
      });

      // Extract callbacks from ImportModel for progress updates
      const { onProgress, onComplete, onError } = importData.callbacks || {};
      const layerName = importData.layerName || 'Combined_LoD3';

      // Upload all files from both folders to backend for combined processing
      try {
        const formData = new FormData();

        // Append all pointcloud files
        const pointcloudFiles = importData.files.pointcloud.files;
        pointcloudFiles.forEach((file, index) => {
          formData.append(`pointcloud_${index}`, file);
        });
        formData.append('pointcloud_count', pointcloudFiles.length.toString());
        formData.append('pointcloud_folder', importData.files.pointcloud.folderName);

        // Append all streetview files
        const streetviewFiles = importData.files.streetview.files;
        streetviewFiles.forEach((file, index) => {
          formData.append(`streetview_${index}`, file);
        });
        formData.append('streetview_count', streetviewFiles.length.toString());
        formData.append('streetview_folder', importData.files.streetview.folderName);

        formData.append('type', 'lod3-data');
        formData.append('layerName', layerName);

        console.log('📤 Uploading folders to backend...');
        console.log(`   ├─ Pointcloud: ${pointcloudFiles.length} files`);
        console.log(`   └─ Streetview: ${streetviewFiles.length} files`);

        // Update progress: uploading
        onProgress?.(5, 'LoD3データをアップロード中...');

        const response = await fetch('/upload-lod3', {
          method: 'POST',
          body: formData
        });

        if (!response.ok) {
          throw new Error(`Upload failed: ${response.statusText}`);
        }

        const result = await response.json();
        console.log('✅ Files uploaded, job created:', result.job_id);

        // Update progress: processing started
        onProgress?.(10, 'LoD3モデルを生成中...');

        // Wait for the combined job to complete with progress updates
        const waitForJob = async (jobId) => {
          while (true) {
            const statusResponse = await fetch(`/jobs/${jobId}/status`);
            const status = await statusResponse.json();
            console.log(`📊 Job ${jobId} status:`, status.status, `(${status.progress}%)`);

            // Update ImportModel progress via callback
            if (onProgress && status.progress !== undefined) {
              onProgress(status.progress);
            }

            if (status.status === 'completed') {
              return status;
            } else if (status.status === 'failed') {
              throw new Error(`Job ${jobId} failed: ${status.error || 'Unknown error'}`);
            }

            await new Promise(resolve => setTimeout(resolve, 3000));
          }
        };

        console.log('⏳ Waiting for LoD3 generation to complete...');
        const job = await waitForJob(result.job_id);
        console.log('✅ LoD3 model generated successfully');

        // The combined OBJ file path
        const objPath = `/outputs/${job.job_id}/${layerName}_lod3.obj`;
        const mtlPath = `/outputs/${job.job_id}/material.mtl`;

        // Create layer for the LoD3 model
        const layerId = addLayer(
          'lod3',
          `LoD3モデル (${importData.files.pointcloud.folderName})`,
          {
            backendGenerated: true,
            objPath: objPath,
            mtlPath: mtlPath,
            jobId: job.job_id,
            url: `${objPath}`
          },
          null,
          {
            originalLayerName: `LoD3モデル (${importData.files.pointcloud.folderName})`,
            format: 'obj',
            backendGenerated: true,
            jobId: job.job_id,
            objPath: objPath,
            createdAt: new Date().toISOString(),
            size: 'Backend processed',
            sourceFiles: {
              pointcloud: importData.files.pointcloud.folderName,
              streetview: importData.files.streetview.folderName
            }
          }
        );

        console.log('🏗️ Created LoD3 layer:', {
          layerId: layerId,
          jobId: job.job_id,
          objUrl: `${objPath}`
        });

        // Hide all layers except LOD3 and satellite/orthophoto
        console.log('🎭 Hiding all layers except LOD3 and satellite images...');

        // Get all layers from LayerManager
        const allLayers = layerManagerRef.current ? layerManagerRef.current.getAllLayers() : [];

        // Find satellite/orthophoto layer IDs
        const satelliteLayerIds = new Set();
        allLayers.forEach((layer) => {
          if (layer.type === 'orthophoto' || layer.type === 'geotiff' || layer.metadata?.format === 'geotiff') {
            satelliteLayerIds.add(layer.id);
            console.log(`   ├─ Keeping satellite layer: ${layer.id}`);
          }
        });

        // Set visible layers to only LOD3 and satellite layers
        const newVisibleLayers = new Set([layerId, ...satelliteLayerIds]);
        setVisibleLayers(newVisibleLayers);
        console.log(`🎭 Visible layers set to:`, Array.from(newVisibleLayers));

        // Hide all other layers in LayerManager
        if (layerManagerRef.current) {
          allLayers.forEach((layer) => {
            const shouldBeVisible = newVisibleLayers.has(layer.id);
            layerManagerRef.current.setLayerVisibility(layer.id, shouldBeVisible);
            console.log(`   ${shouldBeVisible ? '✓' : '✗'} Layer ${layer.id}: ${shouldBeVisible ? 'visible' : 'hidden'}`);
          });
        }

        setAvailableLoDs(prev => new Set([...prev, 'lod3']));
        setProjectStage('lod3-generated');
        console.log(`✅ LoD3 model added and display updated: ${layerId}`);

        // Notify ImportModel of completion
        onComplete?.();

      } catch (error) {
        console.error('❌ LoD3 processing failed:', error);
        // Notify ImportModel of error
        if (onError) {
          onError(error);
        } else {
          alert(`Failed to process LoD3 data: ${error.message}`);
        }
      }

      setShowImportModel(false);
      return;
    }
    
    setShowImportModel(false);
    
    // Detect type by file extension or MIME type
    let type = importData.type;
    if (
      importData.file && (
        importData.file.type === 'image/tiff' ||
        importData.file.name.toLowerCase().endsWith('.tif') ||
        importData.file.name.toLowerCase().endsWith('.tiff')
      )
    ) {
      type = 'geotiff';
    }
    
    if (type === 'geotiff') {
      console.log('📷 Processing GeoTIFF...');
      
      try {
        // Add timeout for GeoTIFF processing to prevent freezing
        const processGeoTIFF = async () => {
          const startTime = Date.now();
          console.log('⏱️ Starting GeoTIFF processing...');
          
          const { rasters, width, height, bbox, colorMap } = await readGeoTiff(importData.file);
          const processingTime1 = Date.now() - startTime;
          console.log(`⏱️ GeoTIFF reading completed in ${processingTime1}ms`);
          
          const pngStartTime = Date.now();
          const pngUrl = rasterToPngUrl(rasters, width, height, colorMap);
          const processingTime2 = Date.now() - pngStartTime;
          console.log(`⏱️ PNG conversion completed in ${processingTime2}ms`);
          
          const coordStartTime = Date.now();
          const coordinates = await getWGS84BoundsFromGeoTiff(importData.file);
          const processingTime3 = Date.now() - coordStartTime;
          console.log(`⏱️ Coordinate transformation completed in ${processingTime3}ms`);
          
          const totalTime = Date.now() - startTime;
          console.log(`⏱️ Total GeoTIFF processing time: ${totalTime}ms`);
          
          return { pngUrl, coordinates, rasters, width, height, bbox, colorMap };
        };
        
        const result = await processGeoTIFF();
        
        // Add to unified layer management system with consistent structure
        const layerId = addLayer(
          'orthophoto',
          importData.layerName,
          {
            rasters: result.rasters,
            width: result.width,
            height: result.height,
            bbox: result.bbox,
            coordinates: result.coordinates,
            previewUrl: result.pngUrl
          },
          null, // No geojson relationship for orthophotos
          {
            ...extractFileMetadata(importData.file),
            originalLayerName: importData.layerName,
            format: 'geotiff',
            bounds: result.coordinates,
            dimensions: { width: result.width, height: result.height },
            bbox: result.bbox,
            hasColorMap: !!result.colorMap,
            bandCount: result.rasters.length
          }
        );
        
        // Add to map for backward compatibility
        // DEBUG: Commented out to prevent orthophoto from actually appearing on the map
        // The layer still appears in the layer panel for toggle preview functionality
        /*
        if (mapInstance && !mapInstance.getSource(layerId)) {
          mapInstance.addSource(layerId, {
            type: 'image',
            url: result.pngUrl,
            coordinates: result.coordinates,
          });
          mapInstance.addLayer({
            id: layerId,
            type: 'raster',
            source: layerId,
            paint: {},
          }, '3d-buildings');
          
          // Register the layer with the layer control system
          if (mapContainerRef.current?.registerExternalLayer) {
            mapContainerRef.current.registerExternalLayer(layerId, layerId);
          }
        }
        */
        
        // Note: No longer adding to customLayers to avoid duplicate entries
        // The layer is already in the unified system and will appear in organized.orthophoto
        
        console.log(`✅ Added GeoTIFF layer: ${layerId}`);
        
      } catch (error) {
        console.error('❌ GeoTIFF processing failed:', error);
        // You could show an error message to the user here
        alert(`Failed to process GeoTIFF: ${error.message}`);
        return;
      }
      
      // NOTE: Removed auto-generation of demo LoD2 model after TIFF upload
      // Users should now manually generate LoD2 models using the backend API
      
      setAvailableLoDs(prev => new Set([...prev, 'lod2'])); // Make LoD2 generation available
      setProjectStage('orthophoto-imported');
      
    } else if (type === 'geojson') {
      console.log('🗺️ Processing GeoJSON...');
      // Handle GeoJSON import
      try {
        const reader = new FileReader();
        reader.onload = (e) => {
          try {
            const rawGeojson = JSON.parse(e.target.result);
            const reprojectedGeojson = reprojectGeojsonToWGS84(rawGeojson);
            
            // Add to unified layer management system
            const layerId = addLayer(
              'geojson',
              getDisplayName(importData.file.name),
              reprojectedGeojson,
              null, // GeoJSON is the root layer, no parent
              {
                ...extractFileMetadata(importData.file),
                originalLayerName: importData.layerName,
                bounds: reprojectedGeojson,
                validation: validateGeojsonData(reprojectedGeojson),
                statistics: getHeightStatistics(reprojectedGeojson),
                jobId: importData.backend?.jobId  // Store backend job ID for linking LoD1/LoD2
              }
            );
            
            // Keep legacy states for backward compatibility
            const newLayer = {
              id: layerId,
              name: getDisplayName(importData.file.name),
              type: 'geojson',
              data: reprojectedGeojson,
              metadata: extractFileMetadata(importData.file)
            };
            
            setImportedGeojsonLayers(prev => [...prev, newLayer]);
            setCurrentGeojsonFile(importData.file);
            setGeojsonData(reprojectedGeojson);
            setShowGeojson(true);

            // Validate the GeoJSON data and log statistics
            const validation = validateGeojsonData(reprojectedGeojson);
            const stats = getHeightStatistics(reprojectedGeojson);
            
            console.log('📊 GeoJSON validation result:', validation);
            console.log('📊 GeoJSON height statistics:', stats);
            
            if (validation.warnings.length > 0) {
              console.warn('⚠️ GeoJSON warnings:', validation.warnings);
            }

            // Update project stage to configuration mode
            setProjectStage('geojson-imported');
            setMapState('configuration');
            
            // Sequence: zoom → show panel → add layer → toggle off/on
            if (mapContainerRef.current?.zoomToGeojson && reprojectedGeojson) {
              setTimeout(() => {
                console.log(`🎯 Step 1: Starting zoom to GeoJSON bounds for layer: ${layerId}`);
                mapContainerRef.current.zoomToGeojson(reprojectedGeojson);
                
                // Step 2: Wait for zoom to complete, then show panel
                setTimeout(() => {
                  console.log(`📋 Step 2: Zoom completed, now showing layer panel`);
                  setShowLayerPanel(true);
                  
                  // Step 3: GeoJSON layer added - user can toggle it on to enter configuration mode
                  setTimeout(() => {
                    console.log(`✅ GeoJSON layer setup completed for: ${layerId}`);
                    console.log(`💡 Toggle the GeoJSON layer ON in the layer panel to enter configuration mode`);
                  }, 300); // Wait after showing panel before adding layer
                }, 1500); // Wait for zoom to complete before showing panel
              }, 50); // Initial delay before starting zoom
            } else {
              // If no zoom function available, skip zoom but follow same sequence
              setTimeout(() => {
                console.log(`� No zoom function, adding layer to visible set: ${layerId}`);
                setShowLayerPanel(true);
                
                // Step 2: GeoJSON layer added - user can toggle it on to enter configuration mode
                setTimeout(() => {
                  console.log(`✅ GeoJSON layer setup completed (no zoom) for: ${layerId}`);
                  console.log(`💡 Toggle the GeoJSON layer ON in the layer panel to enter configuration mode`);
                }, 200);
              }, 200);
            }
            
            console.log(`✅ Added GeoJSON layer: ${layerId}`);
          } catch (parseError) {
            console.error('❌ Failed to parse GeoJSON:', parseError);
          }
        };
        reader.readAsText(importData.file);
      } catch (error) {
        console.error('❌ Failed to read GeoJSON file:', error);
      }
      
    } else if (type === 'orthophoto') {
      console.log('🖼️ Processing orthophoto...');
      const coordinates = [
        [139.6, 35.7], [139.8, 35.7], [139.8, 35.6], [139.6, 35.6]
      ];
      
      // Add to unified layer management system with consistent structure
      const layerId = addLayer(
        'orthophoto',
        importData.layerName,
        {
          previewUrl: importData.previewUrl,
          coordinates
        },
        null, // No geojson relationship for regular orthophotos
        {
          ...(importData.file ? extractFileMetadata(importData.file) : {
            originalName: 'uploaded-image',
            size: 0,
            type: 'image',
            lastModified: new Date().toISOString(),
            uploadedAt: new Date().toISOString()
          }),
          originalLayerName: importData.layerName,
          format: 'image',
          bounds: coordinates,
          source: 'upload'
        }
      );
      
      // Add to map for backward compatibility
      if (mapInstance && !mapInstance.getSource(layerId)) {
        mapInstance.addSource(layerId, {
          type: 'image',
          url: importData.previewUrl,
          coordinates: [
            [139.6, 35.7],
            [139.8, 35.7],
            [139.8, 35.6],
            [139.6, 35.6],
          ],
        });
        mapInstance.addLayer({
          id: layerId,
          type: 'raster',
          source: layerId,
          paint: {},
        }, '3d-buildings');

        // Register the layer with the layer control system
        if (mapContainerRef.current?.registerExternalLayer) {
          mapContainerRef.current.registerExternalLayer(layerId, layerId);
        }
      }
      
      // Note: No longer adding to customLayers to avoid duplicate entries
      // The layer is already in the unified system and will appear in organized.orthophoto
      
      // Update project stage
      setProjectStage('orthophoto-imported');
      setAvailableLoDs(prev => new Set([...prev, 'lod2']));
      
      console.log(`✅ Added orthophoto layer: ${layerId}`);
      
    } else if (type === 'pointcloud') {
      console.log('☁️ Processing point cloud...');
      // Add to unified layer management system with consistent structure
      const layerId = addLayer(
        'pointcloud',
        importData.layerName,
        {
          file: importData.file,
          previewUrl: importData.previewUrl
        },
        null, // Point clouds can be standalone or relate to multiple GeoJSONs
        {
          ...(importData.file ? extractFileMetadata(importData.file) : {
            originalName: 'uploaded-pointcloud',
            size: 0,
            type: 'pointcloud',
            lastModified: new Date().toISOString(),
            uploadedAt: new Date().toISOString()
          }),
          originalLayerName: importData.layerName,
          format: 'pointcloud',
          source: 'upload'
        }
      );
      
      // Update project stage
      setProjectStage('pointcloud-imported');
      setAvailableLoDs(prev => new Set([...prev, 'lod3']));
      
      console.log(`✅ Added point cloud layer: ${layerId}`);
      
      // DEMO MODE: Auto-generate LoD3 model after pointcloud upload
      console.log('🎬 DEMO: Auto-generating LoD3 model...');
      setTimeout(() => {
        const demoLod3Id = addLayer(
          'lod3',
          'Demo LoD3 Model',
          {
            demoMode: true,
            objPath: '/src/assets/test_data/01/lod3/route1/results_route1_lod3_350m.obj',
            mtlPath: '/src/assets/test_data/01/lod3/route1/material.mtl'
          },
          null, // No specific GeoJSON relationship for demo
          {
            objPath: '/src/assets/test_data/01/lod3/route1/results_route1_lod3_350m.obj',
            mtlPath: '/src/assets/test_data/01/lod3/route1/material.mtl',
            generatedFrom: 'demo-workflow',
            lodLevel: 'lod3',
            demoMode: true,
            demoGeneration: true,
            description: 'Demo LoD3 model generated from pointcloud'
          }
        );
        
        setAvailableLoDs(prev => new Set([...prev, 'lod3']));
        setProjectStage('lod3-generated');
        console.log(`🎬 DEMO: Added LoD3 model: ${demoLod3Id}`);
      }, 1500); // Small delay for visual effect
    }
    
    console.log('🏁 Import process completed');

    // Check for backend-generated models after any backend file processing
    if (importData.backend && importData.backend.jobId) {
      console.log('🔍 Checking for backend-generated models after processing...', importData.backend);

      // For GeoJSON uploads, automatically generate LoD1 using frontend logic
      if (importData.type === 'geojson') {
        console.log('🗺️ GeoJSON detected - will automatically generate LoD1 after backend processing completes');

        // Poll backend job until complete, then trigger frontend LoD1 generation
        pollJobAndGenerateLoD1Frontend(importData.backend.jobId);
      }
      // For orthophoto uploads, poll the job status until completion before checking for LoD2 model
      else if (importData.type === 'orthophoto' || importData.file?.type === 'image/tiff') {
        console.log('📸 Orthophoto detected - will poll job status for LoD2 generation');

        // Hide all layers except the orthophoto while generating LoD2
        console.log('🔄 Hiding all layers except orthophoto during LoD2 generation...');
        const orthophotoLayerId = importData.backend?.layerId;
        setVisibleLayers(prev => {
          const layersToKeep = new Set();

          // Keep the orthophoto layer visible
          if (orthophotoLayerId) {
            layersToKeep.add(orthophotoLayerId);
            console.log(`   ├─ Keeping orthophoto layer visible: ${orthophotoLayerId}`);
          }

          // Log hidden layers
          const hiddenCount = prev.size - layersToKeep.size;
          if (hiddenCount > 0) {
            console.log(`   └─ ✅ Hidden ${hiddenCount} layer(s) during generation`);
          }

          return layersToKeep;
        });

        // Explicitly hide GeoJSON Mapbox layers
        console.log(`🔇 Explicitly hiding GeoJSON Mapbox layers during generation...`);
        if (mapInstance && mapInstance.getLayer) {
          const geojsonLayerIds = ['geojson-mask-extrusion', 'geojson-mask-fill', 'geojson-mask-stroke'];
          geojsonLayerIds.forEach(mapLayerId => {
            try {
              if (mapInstance.getLayer(mapLayerId)) {
                mapInstance.setLayoutProperty(mapLayerId, 'visibility', 'none');
                console.log(`   ├─ Hidden GeoJSON layer: ${mapLayerId}`);
              }
            } catch (err) {
              console.warn(`   ├─ Could not hide ${mapLayerId}:`, err);
            }
          });
        }

        pollJobAndCheckLoD2(importData.backend.jobId);
      }
    }
  };
  
  // Function to poll job status and automatically generate LoD1 using frontend logic
  const pollJobAndGenerateLoD1Frontend = async (jobId) => {
    try {
      console.log(`⏳ Polling backend job status for GeoJSON processing: ${jobId}`);

      // Poll job status until completion
      await backendApi.pollJobStatus(
        jobId,
        (status) => {
          // Progress callback
          console.log(`📊 GeoJSON Processing Progress: ${status.progress?.toFixed(0) || 0}%`);
        },
        2000 // Poll every 2 seconds
      );

      console.log(`✅ GeoJSON processing job ${jobId} completed`);
      console.log(`🔨 Now automatically generating LoD1 using frontend logic...`);

      // Wait a bit to ensure all state is updated
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Find the GeoJSON layer that was just imported by jobId
      const geojsonLayer = layerManagerRef.current.getAllLayers().find(
        layer => layer.type === 'geojson' && layer.metadata?.jobId === jobId
      );

      if (!geojsonLayer) {
        console.error(`❌ Could not find GeoJSON layer with jobId: ${jobId}`);
        console.log(`   Available layers:`, layerManagerRef.current.getAllLayers().map(l => ({
          id: l.id,
          type: l.type,
          jobId: l.metadata?.jobId
        })));
        return;
      }

      console.log(`✅ Found GeoJSON layer: ${geojsonLayer.id}`);
      console.log(`   ├─ Layer name: ${geojsonLayer.name}`);
      console.log(`   ├─ Has data: ${!!geojsonLayer.data}`);

      // Get the source GeoJSON data and create LoD1
      const lodData = {
        type: 'FeatureCollection',
        features: geojsonLayer.data.features.map(feature => ({
          ...feature,
          properties: {
            ...feature.properties,
            lodLevel: 'lod1',
            configured: true
          }
        }))
      };

      // Generate name based on source GeoJSON
      const lod1Name = `LoD1モデル (${geojsonLayer.name})`;

      console.log(`🔨 Creating LoD1 layer: ${lod1Name}`);

      // Add to unified layer management system
      const lod1Id = addLayer(
        'lod1',
        lod1Name,
        lodData,
        geojsonLayer.id, // Link to source GeoJSON
        {
          sourceGeojsonFile: geojsonLayer.name,
          generatedFrom: geojsonLayer.data,
          lodLevel: 'lod1',
          configurationData: configurationData, // Store configuration data
          autoGenerated: true,
          jobId: jobId  // Store jobId so we can merge with city assets later
        }
      );

      console.log(`✅ Added LoD1 layer: ${lod1Id}`);

      // Keep legacy state for backward compatibility
      const unifiedLayer = layerManagerRef.current.getLayer(lod1Id);
      const lod1Layer = {
        id: lod1Id,
        name: lod1Name,
        type: 'lod1',
        data: lodData,
        controlId: unifiedLayer?.controlId || lod1Id,
        metadata: {
          sourceGeojsonFile: geojsonLayer.name,
          generatedFrom: geojsonLayer.data,
          lodLevel: 'lod1',
          autoGenerated: true,
          jobId: jobId  // Store jobId so we can merge with city assets later
        }
      };

      setLod1Layers(prev => [...prev, lod1Layer]);

      // Add to available LoDs and update project stage
      setAvailableLoDs(prev => new Set([...prev, 'lod1']));
      setLodData(prev => ({
        ...prev,
        lod1: lodData
      }));

      setProjectStage('lod1-generated');

      // Make the layer visible
      setVisibleLayers(prev => new Set([...prev, lod1Id]));

      console.log(`🎉 LoD1 model generated and displayed automatically!`);

      // Hide GeoJSON layer and exit configuration mode after LoD1 rendering
      console.log(`🔇 Hiding GeoJSON layer and exiting configuration mode...`);

      // Hide GeoJSON from visible layers
      setVisibleLayers(prev => {
        const newVisible = new Set(prev);
        newVisible.delete(geojsonLayer.id);
        return newVisible;
      });

      // Exit configuration mode - set both showGeojson and mapState
      setShowGeojson(false);
      setMapState('normal');

      // Hide Mapbox GeoJSON layers directly
      if (mapInstance && mapInstance.getLayer) {
        const geojsonMapLayerIds = ['geojson-mask-extrusion', 'geojson-mask-fill', 'geojson-mask-stroke'];
        geojsonMapLayerIds.forEach(mapLayerId => {
          try {
            if (mapInstance.getLayer(mapLayerId)) {
              mapInstance.setLayoutProperty(mapLayerId, 'visibility', 'none');
            }
          } catch (err) {
            // Silently ignore if layer doesn't exist
          }
        });
      }

      console.log(`✅ GeoJSON hidden and configuration mode exited`);

      // Check if city assets (roads, vegetation, furniture) were also generated
      await checkForCityAssets(jobId, geojsonLayer);

    } catch (error) {
      console.error(`❌ Error in automatic LoD1 generation:`, error);
      console.error(error.stack);
    }
  };

  // Function to check for and load city assets layer
  const checkForCityAssets = async (jobId, sourceGeojsonLayer) => {
    try {
      console.log(`🌆 Checking for city assets (roads, vegetation, furniture)...`);

      // Get job info to check for city_assets_output
      const jobs = await backendApi.listJobs();
      const job = jobs.find(j => {
        const jId = j.id || j.job_id || j.ID || Object.keys(j).find(key =>
          key.toLowerCase().includes('id') && j[key] && typeof j[key] === 'string'
        );
        return jId === jobId;
      });

      if (!job) {
        console.warn(`⚠️ Job ${jobId} not found`);
        return;
      }

      // Check if city_assets_output exists in job metadata
      if (!job.city_assets_output) {
        console.log(`ℹ️ No city assets output found for job ${jobId}`);
        return;
      }

      console.log(`✅ Found city assets output: ${job.city_assets_output}`);

      // Extract filename from path
      const cityAssetsPath = job.city_assets_output;
      const fileName = cityAssetsPath.split('/').pop();
      const layerName = job?.layer_name || 'Untitled';

      // Construct the URL path
      const cityAssetsObjPath = `/outputs/${jobId}/${fileName}`;
      const cityAssetsMtlPath = `/outputs/${jobId}/material.mtl`;

      // Check if file exists
      const objExists = await backendApi.testObjFileAccess(jobId, fileName);

      if (!objExists) {
        console.warn(`⚠️ City assets file not accessible: ${cityAssetsObjPath}`);
        return;
      }

      console.log(`🎯 City assets file confirmed: ${fileName}`);

      // Check if MTL file exists
      const mtlExists = await backendApi.testObjFileAccess(jobId, 'material.mtl');
      if (mtlExists) {
        console.log(`✅ Material file found: material.mtl`);
      } else {
        console.warn(`⚠️ No material file found, model may not display correctly`);
      }

      // Check if this layer already exists
      const existingLayers = layerManagerRef.current.getLayersByType('lod1');
      const alreadyExists = existingLayers.some(layer =>
        layer.metadata?.jobId === jobId && layer.metadata?.isCityAssets === true
      );

      if (alreadyExists) {
        console.log(`ℹ️ City assets layer already exists for job ${jobId}`);
        return;
      }

      // Find the initial LoD1 layer for this job to merge with
      const initialLod1Layer = existingLayers.find(layer =>
        layer.metadata?.jobId === jobId && layer.metadata?.autoGenerated === true && !layer.metadata?.isCityAssets
      );

      if (!initialLod1Layer) {
        console.warn(`⚠️ No initial LoD1 layer found for job ${jobId}, creating city assets as standalone layer`);
      } else {
        console.log(`🔗 Found initial LoD1 layer to merge with: ${initialLod1Layer.id}`);
      }

      // Add city assets as part of the initial LoD1 layer (merged)
      // They will share the same controlId and toggle together
      const cityAssetsLayerId = addLayer(
        'lod1',
        initialLod1Layer ? initialLod1Layer.name : `LoD1モデル + City Assets (${layerName})`,
        {
          backendGenerated: true,
          objPath: cityAssetsObjPath,
          mtlPath: mtlExists ? cityAssetsMtlPath : null,  // Include MTL path
          jobId: jobId,
          url: `${cityAssetsObjPath}`,
          isCityAssets: true
        },
        sourceGeojsonLayer?.id || null,  // Link to source GeoJSON
        {
          originalLayerName: `City Assets (${layerName})`,
          format: 'obj',
          backendGenerated: true,
          jobId: jobId,
          objPath: cityAssetsObjPath,
          mtlPath: mtlExists ? cityAssetsMtlPath : null,  // Include MTL path in metadata
          createdAt: job?.completed_at || new Date().toISOString(),
          size: 'Backend processed',
          verified: true,
          sourceFile: job.filename,
          sourceJobId: jobId,
          autoGenerated: true,
          isCityAssets: true,  // Mark this as city assets layer
          mergedWithLayer: initialLod1Layer?.id,  // Track which layer this is merged with
          hiddenFromPanel: true  // Hide from layer panel - it's merged with initial LoD1
        }
      );

      console.log(`✅ Added city assets layer: ${cityAssetsLayerId} (merged with ${initialLod1Layer?.id || 'standalone'})`);

      // Get the unified layer and update its controlId to match the initial LoD1 layer
      const cityAssetsUnifiedLayer = layerManagerRef.current.getLayer(cityAssetsLayerId);

      // If we have an initial LoD1 layer, use its controlId so they toggle together
      const sharedControlId = initialLod1Layer?.controlId || cityAssetsUnifiedLayer?.controlId || cityAssetsLayerId;

      if (initialLod1Layer && cityAssetsUnifiedLayer) {
        // Update the city assets layer to use the same controlId as the initial LoD1
        cityAssetsUnifiedLayer.controlId = sharedControlId;
        console.log(`   ├─ Set shared controlId: ${sharedControlId}`);

        // Also update the initial LoD1 layer's name to indicate it includes city assets
        const initialUnifiedLayer = layerManagerRef.current.getLayer(initialLod1Layer.id);
        if (initialUnifiedLayer && !initialUnifiedLayer.name.includes('City Assets')) {
          initialUnifiedLayer.name = `${initialUnifiedLayer.name.replace(')', ' + City Assets)')}`;
          console.log(`   ├─ Updated initial LoD1 name to: ${initialUnifiedLayer.name}`);
        }
      }

      // Also add to lod1Layers state for backward compatibility
      const cityAssetsLegacyLayer = {
        id: cityAssetsLayerId,
        name: initialLod1Layer ? initialLod1Layer.name : `LoD1モデル + City Assets (${layerName})`,
        type: 'lod1',
        data: {
          backendGenerated: true,
          objPath: cityAssetsObjPath,
          mtlPath: mtlExists ? cityAssetsMtlPath : null,
          url: `${cityAssetsObjPath}`,
          isCityAssets: true
        },
        controlId: sharedControlId,  // Use shared controlId
        metadata: {
          isCityAssets: true,
          mergedWithLayer: initialLod1Layer?.id
        }
      };

      setLod1Layers(prev => {
        // Update the initial LoD1 layer's name and controlId if it exists
        if (initialLod1Layer) {
          const updated = prev.map(layer => {
            if (layer.id === initialLod1Layer.id) {
              return {
                ...layer,
                name: layer.name.includes('City Assets') ? layer.name : `${layer.name.replace(')', ' + City Assets)')}`,
                controlId: sharedControlId
              };
            }
            return layer;
          });
          return [...updated, cityAssetsLegacyLayer];
        }
        return [...prev, cityAssetsLegacyLayer];
      });
      console.log(`   ├─ Added to lod1Layers state for rendering`);

      // Make the layer visible (it will automatically show/hide with the initial LoD1 due to shared controlId)
      setVisibleLayers(prev => new Set([...prev, cityAssetsLayerId]));

      console.log(`🎉 City assets merged with initial LoD1 layer - they will toggle together!`);

    } catch (error) {
      console.error(`❌ Error checking for city assets:`, error);
      console.error(error.stack);
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

    // Calculate remaining time based on progress rate
    const msPerPercent = elapsedMs / progressInPhase;
    const remainingProgress = BUILDING_PHASE_END - currentProgress;
    const remainingMs = msPerPercent * remainingProgress;
    const remainingSeconds = Math.ceil(remainingMs / 1000);

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

  // Function to poll job status and check for LoD2 model after completion
  const pollJobAndCheckLoD2 = async (jobId) => {
    try {
      console.log(`⏳ Polling job status for: ${jobId}`);

      // Reset ETA tracking
      buildingPhaseStartTime.current = null;
      buildingPhaseStartProgress.current = null;

      // Show initial status
      setLod2GenerationStatus({ progress: 0, eta: null });

      // Poll job status until completion
      await backendApi.pollJobStatus(
        jobId,
        (status) => {
          // Progress callback
          const currentProgress = status.progress || 0;
          const eta = calculateBuildingETA(currentProgress);
          const etaText = eta ? ` (残り約${eta})` : '';
          console.log(`📊 LoD2 Generation Progress: ${currentProgress.toFixed(0)}%${etaText}`);

          // Update UI state
          setLod2GenerationStatus({ progress: currentProgress, eta: eta });
        },
        3000 // Poll every 3 seconds
      );

      // Clear status when done
      setLod2GenerationStatus(null);

      console.log(`✅ Job ${jobId} completed, checking for LoD2 model...`);

      // Wait a bit for file system to sync
      await new Promise(resolve => setTimeout(resolve, 1000));

      // Now check for the generated LoD2 model
      await checkForGeneratedLoD2Model(jobId);

    } catch (error) {
      console.error(`❌ Error polling job status for ${jobId}:`, error);
      // Still try to check for the model even if polling failed
      await checkForGeneratedLoD2Model(jobId);
    }
  };

  // Function to check for generated LoD2 model for a specific job
  const checkForGeneratedLoD2Model = async (jobId) => {
    try {
      console.log(`🔍 Checking for LoD2 model for job: ${jobId}`);

      // Get job info first to get layer_name
      const jobs = await backendApi.listJobs();
      const job = jobs.find(j => {
        const jId = j.id || j.job_id || j.ID || Object.keys(j).find(key =>
          key.toLowerCase().includes('id') && j[key] && typeof j[key] === 'string'
        );
        return jId === jobId;
      });

      // Construct expected filename using the job's layer_name
      const layerName = job?.layer_name || 'Untitled';
      const expectedFileName = `${layerName}_lod2.obj`;
      const expectedFileNameBMQI = `${layerName}_lod2_bmqi.obj`;

      const possibleFiles = [expectedFileName, 'Untitled_lod2.obj', 'lod2.obj'];

      for (const fileName of possibleFiles) {
        const objExists = await backendApi.testObjFileAccess(jobId, fileName);

        if (objExists) {
          console.log(`🎯 Found generated LoD2 model: ${fileName} for job ${jobId}`);

          // Check if this layer already exists
          const existingLayers = layerManagerRef.current.getLayersByType('lod2');
          const alreadyExists = existingLayers.some(layer =>
            layer.metadata?.jobId === jobId && !layer.metadata?.isBMQI
          );

          if (!alreadyExists) {
            const sourceFilename = job ? job.filename : 'Unknown';
            const lod2ObjPath = `/outputs/${jobId}/${fileName}`;
            const mtlPath = `/outputs/${jobId}/material.mtl`;

            // Add the LoD2 model as a layer (including bbox if available)
            const layerId = addLayer(
              'lod2',
              `LoD2モデル (${sourceFilename})`,
              {
                backendGenerated: true,
                objPath: lod2ObjPath,
                mtlPath: mtlPath,
                jobId: jobId,
                url: `${lod2ObjPath}`,
                bbox: job?.bbox,
                isBMQI: false
              },
              null,
              {
                originalLayerName: `LoD2モデル (${sourceFilename})`,
                format: 'obj',
                backendGenerated: true,
                jobId: jobId,
                objPath: lod2ObjPath,
                createdAt: job?.completed_at || new Date().toISOString(),
                size: 'Backend processed',
                verified: true,
                sourceFile: sourceFilename,
                sourceJobId: jobId,
                autoGenerated: true,
                bbox: job?.bbox,
                isBMQI: false
              }
            );

            // Log bbox info if available
            if (job?.bbox) {
              console.log(`   📍 LoD2 layer has bounding box:`, job.bbox);
            }
            
            console.log(`✅ Auto-added generated LoD2 layer: ${layerId} for job ${jobId} (${fileName})`);

            // Show only the LoD2 layer and orthophoto, hide everything else
            console.log(`🔄 Showing LoD2 and orthophoto, hiding all other layers...`);
            const allLayers = layerManagerRef.current.getAllLayers();

            // Find the orthophoto layer - try by jobId first, then by layer name, then any orthophoto
            let orthophotoLayer = allLayers.find(layer =>
              layer.type === 'orthophoto' && layer.metadata?.jobId === jobId
            );

            // If not found by jobId, try to find by matching layer name
            if (!orthophotoLayer && layerName) {
              orthophotoLayer = allLayers.find(layer =>
                layer.type === 'orthophoto' && layer.name === layerName
              );
            }

            // If still not found, use the most recent orthophoto layer
            if (!orthophotoLayer) {
              const orthophotoLayers = allLayers.filter(layer => layer.type === 'orthophoto');
              if (orthophotoLayers.length > 0) {
                orthophotoLayer = orthophotoLayers[orthophotoLayers.length - 1];
                console.log(`   ├─ Using most recent orthophoto layer: ${orthophotoLayer.id}`);
              }
            }

            setVisibleLayers(prev => {
              const newSet = new Set();

              // Show the new LoD2 layer
              newSet.add(layerId);
              console.log(`   ├─ Showing LoD2 layer: ${layerId}`);

              // Show the orthophoto if found
              if (orthophotoLayer) {
                newSet.add(orthophotoLayer.id);
                console.log(`   ├─ Showing orthophoto layer: ${orthophotoLayer.id}`);
              } else {
                console.log(`   ├─ ⚠️ No orthophoto layer found to show`);
              }

              const hiddenCount = prev.size - newSet.size;
              if (hiddenCount > 0) {
                console.log(`   └─ ✅ Hidden ${hiddenCount} other layer(s)`);
              }

              return newSet;
            });

            // Explicitly hide GeoJSON Mapbox layers
            console.log(`🔇 Explicitly hiding GeoJSON Mapbox layers...`);
            if (mapInstance && mapInstance.getLayer) {
              const geojsonLayerIds = ['geojson-mask-extrusion', 'geojson-mask-fill', 'geojson-mask-stroke'];
              geojsonLayerIds.forEach(mapLayerId => {
                try {
                  if (mapInstance.getLayer(mapLayerId)) {
                    mapInstance.setLayoutProperty(mapLayerId, 'visibility', 'none');
                    console.log(`   ├─ Hidden GeoJSON layer: ${mapLayerId}`);
                  }
                } catch (err) {
                  console.warn(`   ├─ Could not hide ${mapLayerId}:`, err);
                }
              });
            }

            setAvailableLoDs(prev => new Set([...prev, 'lod2']));
            setProjectStage('lod2-generated');

            // Don't return here - we need to check for BMQI model too
          } else {
            console.log(`ℹ️ LoD2 layer already exists for job ${jobId}`);
          }

          break; // Stop checking other files once we find one
        }
      }

      // Also check for BMQI model (this happens after the regular LOD2 check)
      console.log(`🔍 Checking for BMQI LoD2 model for job: ${jobId}`);
      const bmqiObjExists = await backendApi.testObjFileAccess(jobId, expectedFileNameBMQI);

      if (bmqiObjExists) {
        console.log(`🎯 Found generated LoD2 BMQI model: ${expectedFileNameBMQI} for job ${jobId}`);

        // Check if BMQI layer already exists
        const existingLayers = layerManagerRef.current.getLayersByType('lod2');
        const alreadyExists = existingLayers.some(layer =>
          layer.metadata?.jobId === jobId && layer.metadata?.isBMQI
        );

        if (!alreadyExists) {
          const sourceFilename = job ? job.filename : 'Unknown';
          const lod2BmqiObjPath = `/outputs/${jobId}/${expectedFileNameBMQI}`;
          const mtlPath = `/outputs/${jobId}/material.mtl`;

          // Add the BMQI LoD2 model as a separate layer
          const layerId = addLayer(
            'lod2',
            `LoD2モデル (BMQI) (${sourceFilename})`,
            {
              backendGenerated: true,
              objPath: lod2BmqiObjPath,
              mtlPath: mtlPath,
              jobId: jobId,
              url: `${lod2BmqiObjPath}`,
              bbox: job?.bbox,
              isBMQI: true
            },
            null,
            {
              originalLayerName: `LoD2モデル (BMQI) (${sourceFilename})`,
              format: 'obj',
              backendGenerated: true,
              jobId: jobId,
              objPath: lod2BmqiObjPath,
              createdAt: job?.completed_at || new Date().toISOString(),
              size: 'Backend processed',
              verified: true,
              sourceFile: sourceFilename,
              sourceJobId: jobId,
              autoGenerated: true,
              bbox: job?.bbox,
              isBMQI: true
            }
          );

          console.log(`✅ Auto-added generated LoD2 BMQI layer: ${layerId} for job ${jobId}`);

          // Add BMQI layer to visible layers
          setVisibleLayers(prev => new Set([...prev, layerId]));
        } else {
          console.log(`ℹ️ LoD2 BMQI layer already exists for job ${jobId}`);
        }
      } else {
        console.log(`ℹ️ No LoD2 BMQI model found yet for job ${jobId}`);
      }

      if (!bmqiObjExists && !possibleFiles.some(async f => await backendApi.testObjFileAccess(jobId, f))) {
        console.log(`ℹ️ No LoD2 models found yet for job ${jobId}`);
      }
    } catch (error) {
      console.warn('Failed to check for generated LoD2 model:', error);
    }
  };

  const handleCloseImport = useCallback(() => setShowImportModel(false), []);

  // Make checkAndAddExistingLoD2Models available globally for debugging
  if (typeof window !== 'undefined') {
    window.appCheckAndAddLoD2Models = checkAndAddExistingLoD2Models;
    window.appCheckForGeneratedLoD2 = checkForGeneratedLoD2Model;
  }

  async function readGeoTiff(file) {
    const arrayBuffer = await file.arrayBuffer();
    const tiff = await fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const bbox = image.getBoundingBox();
    const rasters = await image.readRasters({ interleave: false }); // [band][pixel]
    const width = image.getWidth();
    const height = image.getHeight();
    const colorMap = image.getFileDirectory().ColorMap || null;
    console.log('Band count:', rasters.length, 'ColorMap:', !!colorMap);
    return { rasters, width, height, bbox, colorMap };
  }

  function rasterToPngUrl(rasters, width, height, colorMap = null) {
    console.log('rasterToPngUrl', { rasters, width, height, colorMap });
    
    const bandAnalysis = rasters.map((band, i) => {
      const firstValues = Array.from(band).slice(0, 10);
      const sampleValues = Array.from(band).slice(0, 1000);
      return {
        band: i,
        length: band.length,
        type: band.constructor.name,
        min: Math.min(...sampleValues),
        max: Math.max(...sampleValues),
        firstValues: firstValues
      };
    });
    
    console.log('📊 Band analysis:', bandAnalysis);
    // Log first values separately for visibility
    bandAnalysis.forEach((analysis, i) => {
      console.log(`📊 Band ${i} first 10 values:`, analysis.firstValues);
    });
    
    if (!Array.isArray(rasters) || !rasters.length) {
      throw new Error('Invalid rasters array');
    }
    if (typeof width !== 'number' || typeof height !== 'number') {
      throw new Error('Invalid width/height');
    }
    if (rasters.some(band => (!Array.isArray(band) && !ArrayBuffer.isView(band)) || band.length !== width * height)) {
      throw new Error('Each band must be an array or TypedArray of length width*height');
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imageData = ctx.createImageData(width, height);

    const bandCount = rasters.length;
    console.log(`🎨 Processing ${bandCount} bands for ${width}x${height} image`);

    // --- Palette handling ---
    let palette = null;
    if (colorMap) {
      const numColors = colorMap.length / 3;
      palette = [];
      for (let i = 0; i < numColors; i++) {
        palette.push([
          colorMap[i] >> 8, // R
          colorMap[i + numColors] >> 8, // G
          colorMap[i + 2 * numColors] >> 8 // B
        ]);
      }
      console.log(`🎨 Using color palette with ${numColors} colors`);
    }

    // Pre-calculate min/max values for normalization to avoid recalculating for each pixel
    let mins = [], maxs = [], ranges = [];
    
    if (bandCount >= 3 && !palette) {
      console.log('📊 Pre-calculating min/max values for multi-band image...');
      
      for (let bandIdx = 0; bandIdx < Math.min(bandCount, 4); bandIdx++) {
        const band = rasters[bandIdx];
        let min = Infinity, max = -Infinity;
        
        // For better accuracy, use more comprehensive sampling
        const sampleSize = Math.min(band.length, 50000); // Increased sample size
        const step = Math.max(1, Math.floor(band.length / sampleSize)); // Ensure step is at least 1
        
        for (let i = 0; i < band.length; i += step) {
          const val = band[i];
          if (typeof val === 'number' && !isNaN(val)) {
            if (val < min) min = val;
            if (val > max) max = val;
          }
        }
        
        // Fallback: if no valid values found, scan first 1000 pixels directly
        if (min === Infinity || max === -Infinity) {
          console.warn(`⚠️ No valid values found in band ${bandIdx}, scanning first 1000 pixels directly`);
          for (let i = 0; i < Math.min(1000, band.length); i++) {
            const val = band[i];
            if (typeof val === 'number' && !isNaN(val)) {
              if (val < min) min = val;
              if (val > max) max = val;
            }
          }
        }
        
        // Use reasonable defaults if still no valid range
        if (min === Infinity) min = 0;
        if (max === -Infinity) max = 255;
        
        mins[bandIdx] = min;
        maxs[bandIdx] = max;
        ranges[bandIdx] = (max - min) || 1;
        
        console.log(`📊 Band ${bandIdx}: min=${min}, max=${max}, range=${ranges[bandIdx]}`);
      }
    }

    const totalPixels = width * height;
    console.log(`🔄 Processing ${totalPixels} pixels...`);
    
    // Process pixels in chunks to prevent blocking
    const processPixels = () => {
      let debugCount = 0;
      for (let i = 0; i < totalPixels; i++) {
        let r = 0, g = 0, b = 0, a = 255;
        
        try {
          if (bandCount === 1 && palette) {
            // Paletted image
            const idx = Math.floor(rasters[0][i]) % palette.length;
            [r, g, b] = palette[idx] || [0, 0, 0];
          } else if (bandCount === 1) {
            // Grayscale
            if (!mins[0]) {
              const band = rasters[0];
              mins[0] = Math.min(...Array.from(band));
              maxs[0] = Math.max(...Array.from(band));
              ranges[0] = (maxs[0] - mins[0]) || 1;
            }
            const val = Math.max(0, Math.min(255, 255 * (rasters[0][i] - mins[0]) / ranges[0]));
            r = g = b = val;
          } else if (bandCount >= 3) {
            // RGB (or RGBA) - handle 3 or 4 band images
            // Ensure we have min/max values for all required bands
            if (mins.length === 0) {
              console.warn('⚠️ Min/max values not pre-calculated, calculating on-demand');
              for (let bandIdx = 0; bandIdx < Math.min(bandCount, 4); bandIdx++) {
                const band = rasters[bandIdx];
                const min = Math.min(...Array.from(band));
                const max = Math.max(...Array.from(band));
                mins[bandIdx] = min;
                maxs[bandIdx] = max;
                ranges[bandIdx] = (max - min) || 1;
                console.log(`📊 On-demand Band ${bandIdx}: min=${min}, max=${max}`);
              }
            }
            
            r = Math.max(0, Math.min(255, 255 * (rasters[0][i] - mins[0]) / ranges[0]));
            g = Math.max(0, Math.min(255, 255 * (rasters[1][i] - mins[1]) / ranges[1]));
            b = Math.max(0, Math.min(255, 255 * (rasters[2][i] - mins[2]) / ranges[2]));
            
            // 🔧 FIX: Ignore alpha channel for 4-band images, always use full opacity
            // Many GeoTIFF images have problematic alpha channels that make images invisible
            a = 255; // Always fully opaque
            
            // Debug first few pixels to see what values we're getting
            if (debugCount < 5) {
              console.log(`🔍 Pixel ${i} debug:`, {
                rawR: rasters[0][i], normalizedR: r,
                rawG: rasters[1][i], normalizedG: g,
                rawB: rasters[2][i], normalizedB: b,
                rawA: bandCount >= 4 ? rasters[3][i] : 255, 
                normalizedA: a, // Always 255 now
                mins: mins.slice(0, 4),
                ranges: ranges.slice(0, 4)
              });
              debugCount++;
            }
          } else if (bandCount === 2) {
            // Two bands: treat first as grayscale, second as alpha or ignore
            if (!mins[0]) {
              mins[0] = Math.min(...Array.from(rasters[0]));
              maxs[0] = Math.max(...Array.from(rasters[0]));
              ranges[0] = (maxs[0] - mins[0]) || 1;
            }
            const val = Math.max(0, Math.min(255, 255 * (rasters[0][i] - mins[0]) / ranges[0]));
            r = g = b = val;
            // Could use second band as alpha if needed
          }
          
          // Ensure values are valid numbers and within range
          r = Math.floor(isNaN(r) ? 0 : Math.max(0, Math.min(255, r)));
          g = Math.floor(isNaN(g) ? 0 : Math.max(0, Math.min(255, g)));
          b = Math.floor(isNaN(b) ? 0 : Math.max(0, Math.min(255, b)));
          a = Math.floor(isNaN(a) ? 255 : Math.max(0, Math.min(255, a)));
          
        } catch (error) {
          console.warn(`❌ Error processing pixel ${i}:`, error);
          r = g = b = 0; // Black pixel for errors
          a = 255;
        }
        
        const pixelIndex = i * 4;
        imageData.data[pixelIndex + 0] = r;
        imageData.data[pixelIndex + 1] = g;
        imageData.data[pixelIndex + 2] = b;
        imageData.data[pixelIndex + 3] = a;
      }
      
      console.log('✅ Pixel processing completed');
      ctx.putImageData(imageData, 0, 0);
      return canvas.toDataURL('image/png');
    };

    return processPixels();
  }

  async function getWGS84BoundsFromGeoTiff(file) {
    console.log('getWGS84BoundsFromGeoTiff called');
    const arrayBuffer = await file.arrayBuffer();
    const tiff = await fromArrayBuffer(arrayBuffer);
    const image = await tiff.getImage();
    const bbox = image.getBoundingBox(); // [minX, minY, maxX, maxY]
    const geoKeys = image.getGeoKeys();

    let fromCRS = 'EPSG:4326'; // Default
    if (geoKeys.ProjectedCSTypeGeoKey) {
      fromCRS = `EPSG:${geoKeys.ProjectedCSTypeGeoKey}`;
    } else if (geoKeys.GeographicTypeGeoKey) {
      fromCRS = `EPSG:${geoKeys.GeographicTypeGeoKey}`;
    }

    console.log('GeoTIFF geoKeys:', geoKeys);
    console.log('Detected CRS:', fromCRS);
    
    // 🔧 FIX: Check if bbox coordinates are already in geographic format
    // If bbox values are in the range of lat/lon (especially for Tokyo area),
    // treat them as already in WGS84 regardless of the reported CRS
    const isGeographicCoordinates = (
      bbox[0] >= 120 && bbox[0] <= 155 && // longitude range for Japan
      bbox[1] >= 20 && bbox[1] <= 50 &&   // latitude range for Japan
      bbox[2] >= 120 && bbox[2] <= 155 &&
      bbox[3] >= 20 && bbox[3] <= 50
    );
    
    if (isGeographicCoordinates) {
      console.log('🔍 Bbox coordinates appear to be geographic (lat/lon), treating as EPSG:4326');
      console.log('📍 Original bbox (appears to be WGS84):', bbox);
      fromCRS = 'EPSG:4326';
    }

    // Define additional Japanese CRS projections
    if (fromCRS === 'EPSG:2451') {
      proj4.defs("EPSG:2451","+proj=tmerc +lat_0=36 +lon_0=139.8333333333333 +k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +units=m +no_defs");
    }
    
    // Add EPSG:6668 definition for GeoTIFF processing
    if (fromCRS === 'EPSG:6668' && !proj4.defs['EPSG:6668']) {
      proj4.defs('EPSG:6668', '+proj=tmerc +lat_0=36 +lon_0=138.5 +k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs');
      console.log('✅ Defined EPSG:6668 for GeoTIFF processing');
    }
    
    // Also add EPSG:6669 as a fallback for Tokyo area data
    if (!proj4.defs['EPSG:6669']) {
      proj4.defs('EPSG:6669', '+proj=tmerc +lat_0=36 +lon_0=139.8333333333333 +k=0.9999 +x_0=0 +y_0=0 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs');
      console.log('✅ Defined EPSG:6669 (CS IX) as fallback for Tokyo area');
    }

    if (fromCRS === 'EPSG:4326') {
      return [
        [bbox[0], bbox[3]], // top-left
        [bbox[2], bbox[3]], // top-right
        [bbox[2], bbox[1]], // bottom-right
        [bbox[0], bbox[1]], // bottom-left
      ];
    }

    const toCRS = 'EPSG:4326';
    
    console.log('🔄 Transforming GeoTIFF coordinates:', {
      fromCRS,
      toCRS,
      originalBbox: bbox
    });
    
    try {
      // Add debug logging for coordinate transformation
      console.log(`📍 Transforming coordinates from ${fromCRS} to ${toCRS}`);
      console.log('📍 Input bbox:', bbox);
      
      const transformedCoords = [
        proj4(fromCRS, toCRS, [bbox[0], bbox[3]]), // top-left
        proj4(fromCRS, toCRS, [bbox[2], bbox[3]]), // top-right
        proj4(fromCRS, toCRS, [bbox[2], bbox[1]]), // bottom-right
        proj4(fromCRS, toCRS, [bbox[0], bbox[1]]), // bottom-left
      ];
      
      console.log('✅ GeoTIFF coordinate transformation successful:', transformedCoords);
      
      // Validate transformed coordinates with more detailed logging
      const validCoords = transformedCoords.every((coord, index) => {
        const isValid = Array.isArray(coord) && 
          coord.length === 2 && 
          typeof coord[0] === 'number' && 
          typeof coord[1] === 'number' &&
          !isNaN(coord[0]) && 
          !isNaN(coord[1]) &&
          coord[0] >= -180 && coord[0] <= 180 &&
          coord[1] >= -90 && coord[1] <= 90;
          
        if (!isValid) {
          console.warn(`❌ Invalid coordinate at index ${index}:`, coord);
        } else {
          console.log(`✅ Valid coordinate ${index}:`, coord);
        }
        
        return isValid;
      });
      
      // Additional validation for Japan region with EPSG:6668 context
      const japanRegionValid = transformedCoords.every(coord => {
        const [lon, lat] = coord;
        return lon >= 120 && lon <= 155 && lat >= 20 && lat <= 50;
      });
      
      if (!validCoords) {
        console.warn('⚠️ Invalid transformed coordinates detected, using fallback bounds');
        console.warn('📍 Original bbox:', bbox);
        console.warn('📍 Failed transformation result:', transformedCoords);
        return [[139.6, 35.7], [139.8, 35.7], [139.8, 35.6], [139.6, 35.6]];
      }
      
      if (!japanRegionValid && fromCRS === 'EPSG:6668') {
        console.warn('⚠️ EPSG:6668 coordinates not in Japan region, trying EPSG:6669 (CS IX) as alternative');
        try {
          const alternativeTransform = [
            proj4('EPSG:6669', toCRS, [bbox[0], bbox[3]]), // top-left
            proj4('EPSG:6669', toCRS, [bbox[2], bbox[3]]), // top-right
            proj4('EPSG:6669', toCRS, [bbox[2], bbox[1]]), // bottom-right
            proj4('EPSG:6669', toCRS, [bbox[0], bbox[1]]), // bottom-left
          ];
          
          const alternativeValid = alternativeTransform.every(coord => {
            const [lon, lat] = coord;
            return Array.isArray(coord) && 
              typeof lon === 'number' && typeof lat === 'number' &&
              !isNaN(lon) && !isNaN(lat) &&
              lon >= 120 && lon <= 155 && lat >= 20 && lat <= 50;
          });
          
          if (alternativeValid) {
            console.log('✅ Alternative EPSG:6669 transformation successful:', alternativeTransform);
            return alternativeTransform;
          }
        } catch (altError) {
          console.warn('⚠️ Alternative EPSG:6669 transformation also failed:', altError);
        }
      }
      
      if (!japanRegionValid) {
        console.warn('⚠️ Coordinates outside Japan region, but using valid WGS84 bounds');
        console.log('📍 Bounds extend beyond typical Japan region, but coordinates are valid');
      } else {
        console.log('✅ Coordinates are within Japan region');
      }
      
      return transformedCoords;
    } catch (error) {
      console.error('❌ GeoTIFF coordinate transformation failed:', error);
      console.log('🔄 Using fallback Tokyo coordinates');
      return [[139.6, 35.7], [139.8, 35.7], [139.8, 35.6], [139.6, 35.6]];
    }
  }

  // function getGeojsonBounds(geojson) {
  //   let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
  //   function processCoords(coords) {
  //     if (typeof coords[0] === 'number') {
  //       const [lon, lat] = coords;
  //       // Only process valid coordinates
  //       if (typeof lon === 'number' && typeof lat === 'number' && 
  //           !isNaN(lon) && !isNaN(lat) && 
  //           lon >= -180 && lon <= 180 && 
  //           lat >= -90 && lat <= 90) {
  //         minLon = Math.min(minLon, lon);
  //         maxLon = Math.max(maxLon, lon);
  //         minLat = Math.min(minLat, lat);
  //         maxLat = Math.max(maxLat, lat);
  //       }
  //     } else {
  //       coords.forEach(processCoords);
  //     }
  //   }
    
  //   if (geojson && geojson.features) {
  //     geojson.features.forEach(f => {
  //       if (f.geometry && f.geometry.coordinates) {
  //         processCoords(f.geometry.coordinates);
  //       }
  //     });
  //   }
    
  //   // Return valid bounds or default to Tokyo area if no valid coordinates found
  //   if (minLon === Infinity || maxLon === -Infinity || minLat === Infinity || maxLat === Infinity) {
  //     console.warn('No valid bounds found, using default Tokyo bounds');
  //     return [[139.6, 35.6], [139.8, 35.8]];
  //   }
    
  //   return [[minLon, minLat], [maxLon, maxLat]];
  // }

  return (
    <div className="app-container">
      {/* Background Map */}
      <MapContainer
        ref={mapContainerRef}
        onMapLoad={handleMapLoad}
        onBuildingClick={handleBuildingClick}
        onPolygonClick={handlePolygonClick}
        selectedBuilding={selectedBuilding}
        mapState={mapState}
        visibleLayers={visibleLayers}
        geojsonData={geojsonData}
        showGeojson={showGeojson}
        lod1Layers={lod1Layers}
        lod3Layers={lod3Layers}
        showHeightColors={showHeightColors}
        organizedLayers={getOrganizedLayers()}
      />

      {/* LoD2 Generation Progress Overlay */}
      {lod2GenerationStatus && (
        <div style={{
          position: 'fixed',
          bottom: '20px',
          left: '50%',
          transform: 'translateX(-50%)',
          backgroundColor: 'rgba(0, 0, 0, 0.85)',
          color: 'white',
          padding: '16px 24px',
          borderRadius: '12px',
          zIndex: 9999,
          minWidth: '280px',
          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.3)'
        }}>
          <div style={{ marginBottom: '8px', fontWeight: '500', fontSize: '14px' }}>
            LoD2モデル生成中...
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <div style={{
              flex: 1,
              height: '8px',
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
              borderRadius: '4px',
              overflow: 'hidden'
            }}>
              <div style={{
                width: `${lod2GenerationStatus.progress}%`,
                height: '100%',
                backgroundColor: lod2GenerationStatus.progress >= 40 && lod2GenerationStatus.progress < 90 ? '#4CAF50' : '#2196F3',
                transition: 'width 0.3s ease'
              }} />
            </div>
            <div style={{ fontSize: '14px', fontWeight: '600', minWidth: '45px' }}>
              {lod2GenerationStatus.progress.toFixed(0)}%
            </div>
          </div>
          {lod2GenerationStatus.eta && (
            <div style={{ marginTop: '8px', fontSize: '12px', color: '#4CAF50' }}>
              残り約{lod2GenerationStatus.eta}
            </div>
          )}
        </div>
      )}

      {/* LoD2 and LoD3 Model Viewers */}
      {mapInstance && (() => {
        // Get all visible LoD2 and LoD3 models from unified layer system
        const organizedLayers = getOrganizedLayers();
        const visibleModels = [];
        
        // Debug: Log current state for troubleshooting
        console.log('🔍 Model Visibility Check:', {
          totalLod2Layers: organizedLayers.lod2?.length || 0,
          totalLod3Layers: organizedLayers.lod3?.length || 0,
          visibleLayerIds: Array.from(visibleLayers),
          visibleLayersCount: visibleLayers.size
        });
        
        // Check for all LoD2 models (demo and backend-generated)
        organizedLayers.lod2?.forEach(layer => {
          console.log(`🔍 Checking LoD2 layer: ${layer.id}`, {
            isVisible: visibleLayers.has(layer.id),
            isDemoMode: !!layer.metadata?.demoMode,
            isBackendGenerated: !!layer.metadata?.backendGenerated,
            metadata: layer.metadata
          });
          
          if (visibleLayers.has(layer.id)) {
            const modelType = layer.metadata?.demoMode ? 'demo' : 
                            layer.metadata?.backendGenerated ? 'backend' : 'other';
            console.log(`✅ Adding visible LoD2 model (${modelType}): ${layer.id}`);
            visibleModels.push({
              ...layer,
              type: 'lod2'
            });
          }
        });
        
        // Check for all LoD3 models (demo and backend-generated)
        organizedLayers.lod3?.forEach(layer => {
          console.log(`🔍 Checking LoD3 layer: ${layer.id}`, {
            isVisible: visibleLayers.has(layer.id),
            isDemoMode: !!layer.metadata?.demoMode,
            isBackendGenerated: !!layer.metadata?.backendGenerated,
            metadata: layer.metadata
          });
          
          if (visibleLayers.has(layer.id)) {
            const modelType = layer.metadata?.demoMode ? 'demo' : 
                            layer.metadata?.backendGenerated ? 'backend' : 'other';
            console.log(`✅ Adding visible LoD3 model (${modelType}): ${layer.id}`);
            visibleModels.push({
              ...layer,
              type: 'lod3'
            });
          }
        });
        
        // console.log(`🎭 Total visible models to render: ${visibleModels.length}`);
        
        return visibleModels.map(layer => (
          <ObjModelViewer
            key={layer.id}
            map={mapInstance}
            objPath={layer.data.demoPath || layer.data.objPath || layer.data.url}
            mtlPath={layer.data.mtlPath}
            showBoundingBox={false}
            onModelLoaded={(data) => {
              const modelType = layer.metadata?.demoMode ? 'Demo' : 
                              layer.metadata?.backendGenerated ? 'Backend' : 'Other';
              console.log(`🎭 ${modelType} ${layer.type.toUpperCase()} Model ${layer.id} loaded successfully:`, data);
            }}
          />
        ));
      })()}

      {/* UI Overlay */}
      <div className="ui-overlay">
        {/* Top Toolbar */}
        <Toolbar
          onToolChange={handleToolChange}
          activeTool={activeTool}
          onImportClick={handleImportClick}
        />

        {/* Left Layer Panel */}
        {showLayerPanel && (
          <LayerPanel
            layers={getOrganizedLayers()}
            onLayerToggle={handleLayerToggle}
            visibleLayers={visibleLayers}
            zoomToLayer={zoomToLayer}
            zoomToOrthophoto={zoomToOrthophoto}
            layerManager={layerManagerRef.current}
          />
        )}

        {/* Top Right Controls */}
        <div className="orthophoto-panel">
          {/* Orthophoto Preview */}
          <OrthophotoPreview 
            mapInstance={mapInstance}
            orthophotoData={(() => {
              // DEMO MODE: Use unified layer system to find orthophoto preview
              const organizedLayers = getOrganizedLayers();
              const visibleOrthophotos = organizedLayers.custom?.filter(l => visibleLayers.has(l.id)) || [];
              if (visibleOrthophotos.length === 0) return null;
              // Use the most recently added orthophoto
              const latestOrthophoto = visibleOrthophotos[visibleOrthophotos.length - 1];
              return latestOrthophoto.previewUrl || latestOrthophoto.data?.previewUrl;
            })()}
          />
        </div>

        <button onClick={() => {
          setShowImportModel(true);
        }}>
          Import Data
        </button>

        {/* Toggle GeoJSON Mask Button */}
        <button onClick={() => setShowGeojson(v => !v)}>
          {showGeojson ? 'Hide' : 'Show'} GeoJSON Mask
        </button>
      </div>

      {/* Models */}
      <ImportModel
        isOpen={showImportModel}
        onClose={handleCloseImport}
        onImport={handleImportComplete}
        projectStage={projectStage}
        availableLoDs={availableLoDs}
      />
      
      {/* Configuration Panel */}
      {mapState === 'configuration' && (
        <ConfigurationPanel
          selectedPolygon={selectedPolygon}
          configurationData={configurationData}
          onConfigurationUpdate={handleConfigurationUpdate}
          onGenerateLoD1={handleGenerateLoD1}
          onClose={handleCloseConfiguration}
        />
      )}

      {/* Debug Panel - only in development */}
      {/* {process.env.NODE_ENV === 'development' && (
        <GeojsonDebugPanel 
          geojsonData={geojsonData} 
          isVisible={geojsonData !== null && import.meta.env.DEV} 
        />
      )} */}
      
      {/* LoD Buttons - Only show in configuration mode after GeoJSON is imported */}
      {mapState === 'configuration' && projectStage !== 'initial' && (
        <LoDButtons
          projectStage={projectStage}
          availableLoDs={availableLoDs}
          onLoDSelect={handleLoDSelect}
        />
      )}
    </div>
  );
}

export default App;