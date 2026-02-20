import React, { useEffect, useRef, useState, forwardRef, useImperativeHandle, useMemo } from 'react';
import mapboxgl from 'mapbox-gl';
import { Threebox } from 'threebox-plugin';
import { processGeojsonHeights } from '../utils/geojsonUtils';
import ObjModelViewer from './ObjModelViewer';

// You'll need to set your Mapbox token here
mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_TOKEN;

const MapContainer = forwardRef(({ onMapLoad, onBuildingClick, onPolygonClick, selectedBuilding, mapState, visibleLayers, geojsonData, showGeojson, zoomGeojsonTrigger, lod1Layers, lod3Layers, showHeightColors, organizedLayers }, ref) => {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const tb = useRef(null); // Threebox instance
  const buildingModel = useRef(null); // Reference to the current 3D model
  const [isLoading, setIsLoading] = useState(true);
  const [showObjModel, setShowObjModel] = useState(false); // Control OBJ model visibility
  const [geojsonCenter, setGeojsonCenter] = useState(null); // Store calculated GeoJSON center
  
  // Store hidden building IDs and filtering state
  const hiddenBuildingIds = useRef(new Set());
  const isFilteringActive = useRef(false);
  
  // Calculate visible LoD3 layers for building restoration logic
  const visibleLod3Layers = useMemo(() => {
    if (!lod3Layers) return new Set();
    return new Set(lod3Layers.filter(layer => visibleLayers.has(layer.id)).map(layer => layer.id));
  }, [lod3Layers, visibleLayers]);

  // Calculate if any OBJ models (LoD2 or LoD3) are visible for unified building hiding logic
  const anyObjModelVisible = useMemo(() => {
    const lod2Visible = showObjModel;
    const lod3Visible = visibleLod3Layers.size > 0;
    
    // Check for LoD2 and LoD3 models from unified layer system
    let unifiedLod2Visible = false;
    let unifiedLod3Visible = false;
    
    if (organizedLayers) {
      // Check LoD2 models (both demo and generated)
      unifiedLod2Visible = organizedLayers.lod2?.some(layer => 
        visibleLayers.has(layer.id)
      ) || false;
      
      // Check LoD3 models (both demo and generated)
      unifiedLod3Visible = organizedLayers.lod3?.some(layer => 
        visibleLayers.has(layer.id)
      ) || false;
    }
    
    return lod2Visible || lod3Visible || unifiedLod2Visible || unifiedLod3Visible;
  }, [showObjModel, visibleLod3Layers, organizedLayers, visibleLayers]);
  
  // Layer registry to track what layers belong to what controls
  const layerRegistry = useRef(new Map());
  
  // Layer state memory - remembers visibility state when layers are out of view
  const layerStateMemory = useRef(new Map());
  
  // Track layer bounds for bound-aware control
  const layerBounds = useRef(new Map());

  // Check if a layer is a Mapbox basemap layer that should never be controlled
  const isMapboxBasemapLayer = (layerId) => {
    // Mapbox basemap layers typically include these prefixes/patterns
    const basemapPatterns = [
      'road', 'bridge', 'tunnel', 'ferry', 'aeroway',
      'building', 'water', 'waterway', 'landuse', 'landcover',
      'admin', 'boundary', 'poi', 'transit', 'place',
      'natural', 'park', 'hillshade', 'contour',
      'settlement', 'state', 'country', 'continent',
      'land', 'background', 'sky', 'hillshade'
    ];

    // Also check for specific layer names
    const specificLayers = ['3d-buildings', 'geojson-mask-extrusion', 'geojson-mask-stroke', 'obj-center-marker'];

    if (specificLayers.includes(layerId)) return true;

    // Check if layer ID starts with any basemap pattern
    return basemapPatterns.some(pattern => layerId.toLowerCase().startsWith(pattern));
  };

  // Register a layer with its control source
  const registerLayer = (layerId, controlId, layerType = 'standard', bounds = null) => {
    // NEVER register Mapbox basemap layers
    if (isMapboxBasemapLayer(layerId)) {
      return;
    }

    // Check if layer is already registered
    const existing = layerRegistry.current.get(layerId);
    if (existing && existing.controlId === controlId) {
      return; // Already registered correctly
    }

    layerRegistry.current.set(layerId, {
      controlId,
      layerType,
      timestamp: Date.now()
    });

    // Store bounds if provided
    if (bounds) {
      layerBounds.current.set(layerId, bounds);
    }
  };
  
  // Unregister a layer
  const unregisterLayer = (layerId) => {
    layerRegistry.current.delete(layerId);
    layerBounds.current.delete(layerId);
    layerStateMemory.current.delete(layerId);
  };
  
  // Remember layer visibility state
  const rememberLayerState = (layerId, isVisible) => {
    layerStateMemory.current.set(layerId, isVisible);
  };
  
  // Get remembered layer state
  const getRememberedState = (layerId) => {
    return layerStateMemory.current.get(layerId) || false;
  };
  
  // Check if layer is within current map bounds
  const isLayerInCurrentView = (layerId) => {
    if (!map.current) return false;
    
    const mapBounds = map.current.getBounds();
    const layerBound = layerBounds.current.get(layerId);
    
    if (!layerBound) {
      // If no bounds stored, assume it could be visible
      return true;
    }
    
    // Check if layer bounds intersect with current map bounds
    const layerBounds_coords = layerBound;
    return mapBounds.intersects(layerBounds_coords);
  };
  
  // Get layers controlled by a specific control ID
  const getLayersForControl = (controlId) => {
    const layers = [];
    for (const [layerId, info] of layerRegistry.current.entries()) {
      if (info.controlId === controlId) {
        layers.push(layerId);
      }
    }
    return layers;
  };
  
  // Check if a layer should be controlled by the visibility system
  const shouldControlLayer = (layerId, controlId) => {
    const registration = layerRegistry.current.get(layerId);
    if (!registration) {
      console.warn(`⚠️ Layer "${layerId}" not found in registry`);
      return false;
    }
    return registration.controlId === controlId;
  };
  
  // Safely control layer visibility with bounds checking
  const setLayerVisibility = (layerId, isVisible) => {
    // NEVER control Mapbox basemap layers
    if (isMapboxBasemapLayer(layerId)) {
      return;
    }

    if (!map.current) {
      rememberLayerState(layerId, isVisible);
      return;
    }

    if (!map.current.getLayer(layerId)) {
      rememberLayerState(layerId, isVisible);
      return;
    }

    try {
      map.current.setLayoutProperty(layerId, 'visibility', isVisible ? 'visible' : 'none');
      rememberLayerState(layerId, isVisible);
    } catch (error) {
      console.error(`Error setting visibility for layer ${layerId}:`, error);
      rememberLayerState(layerId, isVisible);
    }
  };

  useEffect(() => {
    if (map.current) return; // Initialize map only once

    map.current = new mapboxgl.Map({
      container: mapContainer.current,
      style: 'mapbox://styles/mapbox/light-v11', // Light theme for architecture
      center: [139.7, 35.7], // Tokyo coordinates - adjust for your city
      zoom: 16,
      pitch: 60, // 3D perspective
      bearing: 0,
      antialias: true
    });
    
    map.current.on('load', () => {
      setIsLoading(false);
      
      // Initialize Threebox
      window.tb = tb.current = new Threebox(
        map.current,
        map.current.getCanvas().getContext('webgl'),
        {
          defaultLights: true,
          enableSelectingObjects: false,
          enableDraggingObjects: false,
          enableRotatingObjects: false,
          enableTooltips: false
        }
      );
      
      // Add 3D buildings layer with initial color based on showHeightColors prop
      map.current.addLayer({
        id: '3d-buildings',
        source: 'composite',
        'source-layer': 'building',
        filter: ['==', 'extrude', 'true'],
        type: 'fill-extrusion',
        minzoom: 15,
        paint: {
          'fill-extrusion-color': showHeightColors ? [
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            '#8A2BE2', // Purple when selected
            [
              'interpolate',
              ['linear'],
              ['get', 'height'],
              0, '#2196F3',    // 0-10m: Blue
              10, '#4CAF50',   // 10-20m: Green
              20, '#FFEB3B',   // 20-30m: Yellow
              30, '#FF9800',   // 30-45m: Orange
              45, '#F44336'    // 45m+: Red
            ]
          ] : [
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            '#8A2BE2', // Purple when selected
            '#ffffff'  // White for normal buildings
          ],
          'fill-extrusion-height': [
            'interpolate',
            ['linear'],
            ['zoom'],
            15, 0,
            15.05, ['get', 'height']
          ],
          'fill-extrusion-base': [
            'interpolate',
            ['linear'],
            ['zoom'],
            15, 0,
            15.05, ['get', 'min_height']
          ],
          'fill-extrusion-opacity': showHeightColors ? 1 : 0.8
        }
      });

      // Background building clicks are completely disabled
      // Other map interactions (pan, zoom, rotate) remain enabled
      map.current.on('click', '3d-buildings', (e) => {
        // Click handler disabled - do nothing
        return;
      });

      // Cursor no longer changes on hover since buildings are not clickable
      map.current.on('mouseenter', '3d-buildings', () => {
        // Keep default cursor
        return;
      });

      map.current.on('mouseleave', '3d-buildings', () => {
        // Keep default cursor
        return;
      });

      onMapLoad && onMapLoad(map.current);
    });

    // Add navigation controls
    map.current.addControl(new mapboxgl.NavigationControl(), 'bottom-left');

    return () => {
      if (map.current) {
        map.current.remove();
        map.current = null;
      }
    };
  }, []);

  // Update selected building highlight
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;

    // Remove previous selection
    map.current.removeFeatureState({
      source: 'composite',
      sourceLayer: 'building'
    });

    // Add new selection
    if (selectedBuilding) {
      map.current.setFeatureState({
        source: 'composite',
        sourceLayer: 'building',
        id: selectedBuilding.id
      }, {
        selected: true
      });
    }
  }, [selectedBuilding]);

  // Update building colors based on height color toggle only
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;
    
    try {
      // Use the centralized building visibility function
      updateBuildingVisibility();
    } catch (error) {
      console.log('Building layer not ready yet');
    }
  }, [showHeightColors]); // Only depend on showHeightColors toggle

  // DEDICATED OBJ Model building hiding logic - Direct trigger for LoD2/LoD3 models
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;

    // Get footprint data for visible OBJ models
    let objModelFootprintData = { type: 'FeatureCollection', features: [] };

    // Check for LoD2 models from unified layer system
    if (organizedLayers && organizedLayers.lod2) {
      organizedLayers.lod2.forEach(layer => {
        if (visibleLayers.has(layer.id)) {
          // Find the source GeoJSON layer with the same job ID
          const sourceGeojsonLayer = organizedLayers.geojson?.find(geojsonLayer => {
            return geojsonLayer.geojsonId === layer.geojsonId ||
                   geojsonLayer.id === layer.geojsonId ||
                   (geojsonLayer.metadata?.jobId && layer.metadata?.jobId &&
                    geojsonLayer.metadata.jobId === layer.metadata.jobId);
          });

          if (sourceGeojsonLayer && sourceGeojsonLayer.data) {
            const processedData = processGeojsonHeights(sourceGeojsonLayer.data);
            if (processedData.features) {
              objModelFootprintData.features.push(...processedData.features);
            }
          } else if (layer.data) {
            const processedData = processGeojsonHeights(layer.data);
            if (processedData.features) {
              objModelFootprintData.features.push(...processedData.features);
            }
          }
        }
      });
    }

    // Check for LoD3 models from unified layer system
    if (organizedLayers && organizedLayers.lod3) {
      organizedLayers.lod3.forEach(layer => {
        if (visibleLayers.has(layer.id)) {
          const sourceGeojsonLayer = organizedLayers.geojson?.find(geojsonLayer => {
            return geojsonLayer.geojsonId === layer.geojsonId ||
                   geojsonLayer.id === layer.geojsonId ||
                   (geojsonLayer.metadata?.jobId && layer.metadata?.jobId &&
                    geojsonLayer.metadata.jobId === layer.metadata.jobId);
          });

          if (sourceGeojsonLayer && sourceGeojsonLayer.data) {
            const processedData = processGeojsonHeights(sourceGeojsonLayer.data);
            if (processedData.features) {
              objModelFootprintData.features.push(...processedData.features);
            }
          } else if (layer.data) {
            const processedData = processGeojsonHeights(layer.data);
            if (processedData.features) {
              objModelFootprintData.features.push(...processedData.features);
            }
          }
        }
      });
    }

    // Legacy support: also check geojsonData for backward compatibility
    const hasLegacyGeojsonData = geojsonData && geojsonData.features && geojsonData.features.length > 0;
    if (anyObjModelVisible && hasLegacyGeojsonData && objModelFootprintData.features.length === 0) {
      const processedGeoJsonData = processGeojsonHeights(geojsonData);
      if (processedGeoJsonData.features) {
        objModelFootprintData.features.push(...processedGeoJsonData.features);
      }
    }

    // If OBJ models are visible and we have footprint data, hide buildings
    if (anyObjModelVisible && objModelFootprintData.features.length > 0) {
      // Use setTimeout to ensure the models are fully loaded before hiding buildings
      setTimeout(() => {
        hideBuildingsInGeojsonArea(objModelFootprintData);
        isFilteringActive.current = true;
      }, 300);
    }
    // If no OBJ models are visible but filtering was active, restore buildings
    else if (!anyObjModelVisible && isFilteringActive.current) {
      // Check if other models (GeoJSON, LoD1) are still visible before restoring
      const geojsonVisible = showGeojson || visibleLayers.has('Geojson-footprint');
      const anyLod1Visible = lod1Layers && lod1Layers.some(layer => visibleLayers.has(layer.id));

      if (!geojsonVisible && !anyLod1Visible) {
        clearBuildingFilters();
        isFilteringActive.current = false;
      }
    }
  }, [anyObjModelVisible, geojsonData, showGeojson, visibleLayers, lod1Layers, organizedLayers]);

  // UNIFIED building hiding logic for OTHER model types (GeoJSON, LoD1)
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded()) return;

    // Check what data sources are available and visible
    const anyLod1Visible = lod1Layers && lod1Layers.some(layer => visibleLayers.has(layer.id));
    const hasGeojsonData = geojsonData && geojsonData.features && geojsonData.features.length > 0;
    const geojsonVisible = showGeojson || visibleLayers.has('Geojson-footprint');

    // Collect all footprint data from visible sources (excluding OBJ models)
    let combinedFootprintData = { type: 'FeatureCollection', features: [] };
    let anyNonObjModelVisible = false;

    // Add GeoJSON data if visible and available
    if (hasGeojsonData && geojsonVisible) {
      const processedGeoJsonData = processGeojsonHeights(geojsonData);
      if (processedGeoJsonData.features) {
        combinedFootprintData.features.push(...processedGeoJsonData.features);
        anyNonObjModelVisible = true;
      }
    }

    // Add LoD1 data if visible and available
    if (anyLod1Visible && lod1Layers) {
      lod1Layers.forEach(layer => {
        const shouldShow = visibleLayers.has(layer.id);
        if (shouldShow && layer.data) {
          const processedData = processGeojsonHeights(layer.data);
          if (processedData.features) {
            combinedFootprintData.features.push(...processedData.features);
            anyNonObjModelVisible = true;
          }
        }
      });
    }

    // Apply building hiding if we have any footprint data and non-OBJ models are visible
    if (combinedFootprintData.features.length > 0 && anyNonObjModelVisible) {
      hideBuildingsInGeojsonArea(combinedFootprintData);
      isFilteringActive.current = true;
    }
    // Only restore buildings if no models are visible at all (including OBJ models)
    else if (!anyNonObjModelVisible && !anyObjModelVisible && isFilteringActive.current) {
      clearBuildingFilters();
      isFilteringActive.current = false;
    }
  }, [geojsonData, lod1Layers, visibleLayers, showGeojson, anyObjModelVisible]);

  // Helper function to safely add layers without conflicts and register them
  const safelyAddLayer = (layerConfig, sourceConfig = null, controlId = null, layerType = 'standard') => {
    if (!map.current) return false;
    
    try {
      // Remove existing layer if it exists
      if (map.current.getLayer(layerConfig.id)) {
        console.log(`Removing existing layer: ${layerConfig.id}`);
        map.current.removeLayer(layerConfig.id);
        unregisterLayer(layerConfig.id);
      }
      
      // Remove existing source if it exists and new source provided
      if (sourceConfig && map.current.getSource(sourceConfig.id)) {
        console.log(`Removing existing source: ${sourceConfig.id}`);
        map.current.removeSource(sourceConfig.id);
      }
      
      // Add new source if provided
      if (sourceConfig) {
        map.current.addSource(sourceConfig.id, sourceConfig.config);
        console.log(`Added source: ${sourceConfig.id}`);
      }
      
      // Add the layer
      map.current.addLayer(layerConfig);

      // Register the layer with its control
      if (controlId) {
        registerLayer(layerConfig.id, controlId, layerType);
      }

      return true;
    } catch (error) {
      console.error(`Error adding layer ${layerConfig.id}:`, error);
      return false;
    }
  };

  // Debug utility to list all layers on the map
  const debugListAllLayers = () => {
    if (!map.current) return;
    
    const mapLayers = map.current.getStyle().layers || [];
    console.log('=== ALL LAYERS ON MAP ===');
    mapLayers.forEach((layer, index) => {
      const visibility = map.current.getLayoutProperty(layer.id, 'visibility') || 'visible';
      const registration = layerRegistry.current.get(layer.id);
      const controlInfo = registration ? `controlled by: ${registration.controlId}` : 'unregistered';
      console.log(`${index + 1}. "${layer.id}" (${layer.type}) - ${visibility} (${controlInfo})`);
    });
    console.log('=== LAYER REGISTRY ===');
    layerRegistry.current.forEach((info, layerId) => {
      console.log(`"${layerId}" → control: "${info.controlId}", type: ${info.layerType}`);
    });
    console.log('========================');
  };

  // Function to register existing layers that were added externally
  const registerExternalLayer = (layerId, controlId) => {
    if (map.current && map.current.getLayer(layerId)) {
      registerLayer(layerId, controlId, 'external');
      console.log(`🔗 Registered external layer "${layerId}" with control "${controlId}"`);
      return true;
    } else {
      console.warn(`Cannot register external layer "${layerId}" - layer not found on map`);
      return false;
    }
  };

  // Function to auto-register unregistered layers when they appear
  const autoRegisterUnknownLayers = () => {
    if (!map.current) return;

    const mapLayers = map.current.getStyle().layers || [];

    mapLayers.forEach(layer => {
      // Skip Mapbox basemap layers
      if (isMapboxBasemapLayer(layer.id)) {
        return;
      }

      if (layerRegistry.current.has(layer.id)) {
        return;
      }

      // Auto-register unknown layers with their own ID as control
      registerLayer(layer.id, layer.id, 'auto-detected');
    });
  };

  // Expose debug and registration functions globally for testing
  useEffect(() => {
    if (typeof window !== 'undefined') {
      window.debugListAllLayers = debugListAllLayers;
      window.registerExternalLayer = registerExternalLayer;
      window.autoRegisterUnknownLayers = autoRegisterUnknownLayers;
      
      // Add building state debugging functions
      window.debugBuildingState = () => {
        console.log('=== BUILDING STATE DEBUG ===');
        console.log('Hidden building IDs:', Array.from(hiddenBuildingIds.current));
        console.log('Filtering active:', isFilteringActive.current);
        console.log('Has GeoJSON data:', !!(geojsonData && geojsonData.features && geojsonData.features.length > 0));
        console.log('Show GeoJSON:', showGeojson);
        console.log('Show Height Colors:', showHeightColors);
        console.log('Map State:', mapState);
        console.log('Visible layers:', Array.from(visibleLayers));
        console.log('--- LoD1 Layers ---');
        console.log('LoD1 layers available:', !!lod1Layers);
        if (lod1Layers) {
          lod1Layers.forEach(layer => {
            const isVisible = visibleLayers.has(layer.id);
            const featureCount = layer.data?.features?.length || 0;
            console.log(`  ${layer.id}: visible=${isVisible}, features=${featureCount}`);
          });
        }
        console.log('--- OBJ Model States ---');
        console.log('LoD2 (showObjModel):', showObjModel);
        console.log('LoD3 layers visible:', Array.from(visibleLod3Layers));
        console.log('Any OBJ model visible (unified):', anyObjModelVisible);
        
        if (map.current && map.current.getLayer('3d-buildings')) {
          const visibility = map.current.getLayoutProperty('3d-buildings', 'visibility') || 'visible';
          const color = map.current.getPaintProperty('3d-buildings', 'fill-extrusion-color');
          const opacity = map.current.getPaintProperty('3d-buildings', 'fill-extrusion-opacity');
          console.log('3D buildings layer visibility:', visibility);
          console.log('3D buildings color config:', color);
          console.log('3D buildings opacity:', opacity);
        }
        console.log('========================');
      };
      
      window.forceBuildingRestore = () => {
        console.log('🚨 Force restoring buildings...');
        clearBuildingFilters();
        isFilteringActive.current = false;
      };

      window.debugObjModelState = () => {
        console.log('=== OBJ MODEL STATE DEBUG ===');
        console.log('showObjModel (LoD2):', showObjModel);
        console.log('visibleLod3Layers:', Array.from(visibleLod3Layers));
        console.log('anyObjModelVisible (unified):', anyObjModelVisible);
        console.log('Generated-example_lod2 control visible:', visibleLayers.has('Generated-example_lod2'));
        console.log('LoD3 controls visible:', lod3Layers ? lod3Layers.map(layer => ({ id: layer.id, visible: visibleLayers.has(layer.id) })) : 'No LoD3 layers');
        console.log('============================');
      };

      window.debugLod1State = () => {
        console.log('=== LoD1 STATE DEBUG ===');
        console.log('LoD1 layers available:', !!lod1Layers);
        console.log('LoD1 layers count:', lod1Layers ? lod1Layers.length : 0);
        
        if (lod1Layers) {
          lod1Layers.forEach((layer, index) => {
            const isVisible = visibleLayers.has(layer.id);
            const featureCount = layer.data?.features?.length || 0;
            const hasData = !!(layer.data && layer.data.features);
            console.log(`LoD1 Layer ${index + 1}:`);
            console.log(`  ├─ ID: ${layer.id}`);
            console.log(`  ├─ Visible: ${isVisible}`);
            console.log(`  ├─ Has data: ${hasData}`);
            console.log(`  ├─ Feature count: ${featureCount}`);
            if (hasData && layer.data.features.length > 0) {
              const firstFeature = layer.data.features[0];
              console.log(`  └─ First feature properties:`, firstFeature.properties);
            }
          });
        }
        
        const anyLod1Visible = lod1Layers && lod1Layers.some(layer => visibleLayers.has(layer.id));
        console.log('Any LoD1 visible:', anyLod1Visible);
        console.log('=======================');
      };
      
      console.log('Debug helpers available:');
      console.log('  - window.debugListAllLayers() - Show all layers and registrations');
      console.log('  - window.registerExternalLayer(layerId, controlId) - Register imported layers');
      console.log('  - window.autoRegisterUnknownLayers() - Auto-register unregistered layers');
      console.log('  - window.debugBuildingState() - Show current building state');
      console.log('  - window.debugObjModelState() - Show OBJ model states (LoD2/LoD3)');
      console.log('  - window.debugLod1State() - Show LoD1 layer states and data');
      console.log('  - window.forceBuildingRestore() - Force restore all hidden buildings');
    }
  }, []);

  // Centralized layer visibility management with strict control binding
  useEffect(() => {
    if (!map.current || !map.current.isStyleLoaded() || !visibleLayers) return;

    // Auto-register any unknown layers first
    autoRegisterUnknownLayers();

    // Get all layers currently on the map
    const mapLayers = map.current.getStyle().layers || [];
    const existingLayerIds = mapLayers.map(layer => layer.id);

    // Process visible controls
    Array.from(visibleLayers).forEach(layerId => {
      // Get the unified layer to find its controlId
      const unifiedLayer = organizedLayers?.geojson?.find(l => l.id === layerId) ||
                          organizedLayers?.lod1?.find(l => l.id === layerId) ||
                          organizedLayers?.lod2?.find(l => l.id === layerId) ||
                          organizedLayers?.lod3?.find(l => l.id === layerId) ||
                          organizedLayers?.orthophoto?.find(l => l.id === layerId);

      const controlId = unifiedLayer?.controlId || layerId;

      // Get all layers that belong to this EXACT control
      const controlledLayers = getLayersForControl(controlId);

      // Verify each layer actually belongs to this control
      controlledLayers.forEach(mapboxLayerId => {
        const registration = layerRegistry.current.get(mapboxLayerId);

        if (!registration || registration.controlId !== controlId) {
          return;
        }

        // This layer definitely belongs to this control
        if (existingLayerIds.includes(mapboxLayerId)) {
          setLayerVisibility(mapboxLayerId, true);
        } else {
          rememberLayerState(mapboxLayerId, true);
        }
      });

      // Handle special case for OBJ model creation
      if (controlId === 'Generated-example_lod2' && controlledLayers.length === 0) {
        loadAndDisplayObjModel();
      }

      // Handle case where control matches a layer name directly
      if (controlledLayers.length === 0 && existingLayerIds.includes(controlId)) {
        registerLayer(controlId, controlId, 'auto-matched');
        setLayerVisibility(controlId, true);
      }
    });

    // Hide layers whose controls are NOT in visibleLayers
    layerRegistry.current.forEach((info, layerId) => {
      // Skip Mapbox basemap layers (safety check)
      if (isMapboxBasemapLayer(layerId)) {
        return;
      }

      const shouldBeVisible = visibleLayers.has(info.controlId);

      if (!shouldBeVisible) {
        setLayerVisibility(layerId, false);
      } else {
        setLayerVisibility(layerId, true);
      }
    });

    // Handle OBJ model removal if its control is not visible
    const objControlVisible = visibleLayers.has('Generated-example_lod2');

    // Update OBJ model visibility based on control state
    if (objControlVisible && !showObjModel) {
      setShowObjModel(true);
    } else if (!objControlVisible && showObjModel) {
      setShowObjModel(false);
    }
  }, [visibleLayers, anyObjModelVisible]);

  useEffect(() => {
    console.log('🎬 GeoJSON useEffect triggered:', {
      hasMap: !!map.current,
      styleLoaded: map.current?.isStyleLoaded(),
      showGeojson,
      mapState,
      hasGeojsonData: !!geojsonData,
      visibleLayersSize: visibleLayers?.size,
      hasOrganizedLayers: !!organizedLayers
    });

    if (!map.current) {
      console.log('⏸️ GeoJSON useEffect: Map not available, returning early');
      return;
    }

    // Helper function to encapsulate the GeoJSON rendering logic
    const renderGeoJsonLayers = () => {
      if (!map.current) {
        console.log('⏸️ renderGeoJsonLayers: Map not available');
        return;
      }

      console.log('🎨 renderGeoJsonLayers: Starting to render GeoJSON layers');

    const sourceId = 'geojson-mask';
    const extrusionLayerId = 'geojson-mask-extrusion';
    const fillLayerId = 'geojson-mask-fill';
    const strokeLayerId = 'geojson-mask-stroke';

    // Check if layers already exist - if so, just update visibility instead of recreating everything
    const layerExists = map.current.getLayer(fillLayerId);
    const sourceExists = map.current.getSource(sourceId);

    if (layerExists && sourceExists && (showGeojson || mapState === 'configuration') && geojsonData) {
      console.log('🔄 GeoJSON layers already exist, updating visibility only');

      // Determine visibility based on organized layers
      let shouldShowGeojsonLayers = false;
      if (showGeojson || mapState === 'configuration') {
        shouldShowGeojsonLayers = true;
      }

      const visibility = shouldShowGeojsonLayers ? 'visible' : 'none';
      console.log(`   ├─ Setting visibility to: ${visibility}`);

      // Update visibility for all GeoJSON layers
      const allLayers = [fillLayerId, strokeLayerId, extrusionLayerId];
      allLayers.forEach(layerId => {
        if (map.current.getLayer(layerId)) {
          map.current.setLayoutProperty(layerId, 'visibility', visibility);
        }
      });

      // Don't recreate layers - exit early to preserve feature state
      return;
    }

    console.log('🎨 Creating GeoJSON layers for the first time');

    // Remove previous layers if they exist (only when recreating)
    const layersToRemove = [extrusionLayerId, fillLayerId, strokeLayerId];
    layersToRemove.forEach(layer => {
      if (map.current.getLayer(layer)) {
        map.current.removeLayer(layer);
      }
    });
    if (map.current.getSource(sourceId)) {
      map.current.removeSource(sourceId);
    }

    if ((showGeojson || mapState === 'configuration') && geojsonData) {
      // Find the active GeoJSON layer ID from visibleLayers for proper registration
      let activeGeojsonLayerId = null;
      let shouldShowGeojsonLayers = false;

      // If showGeojson is explicitly true OR we're in configuration mode, we should show the layers
      // This is the primary visibility control
      if (showGeojson || mapState === 'configuration') {
        shouldShowGeojsonLayers = true;
        console.log(`📍 GeoJSON should be visible (showGeojson: ${showGeojson}, mapState: ${mapState})`);
      }

      // Try to find the active GeoJSON layer ID from organizedLayers for registration
      if (organizedLayers && organizedLayers.geojson) {
        const visibleGeojsonLayer = organizedLayers.geojson.find(layer => visibleLayers.has(layer.id));
        if (visibleGeojsonLayer) {
          activeGeojsonLayerId = visibleGeojsonLayer.id;
          console.log(`📍 Active GeoJSON layer ID: ${activeGeojsonLayerId} (from unified system)`);
        } else {
          console.log(`📍 No visible GeoJSON layer found in organizedLayers, will use fallback`);
        }
      }

      // Fallback to legacy ID if no layer found (for backward compatibility)
      if (!activeGeojsonLayerId) {
        activeGeojsonLayerId = 'footprint-example';
        console.log(`📍 Using fallback GeoJSON layer ID: ${activeGeojsonLayerId}`);
      }

      // Process GeoJSON data to ensure it has proper height properties for extrusion
      const processedGeoJsonData = processGeojsonHeights(geojsonData);

      // Add feature IDs for feature state support (needed for polygon highlighting)
      if (processedGeoJsonData.features) {
        processedGeoJsonData.features.forEach((feature, index) => {
          feature.id = index;
        });
      }

      // Add GeoJSON source for masking OSM buildings
      map.current.addSource(sourceId, {
        type: 'geojson',
        data: processedGeoJsonData
      });
      
      // Add the GeoJSON source for filtering OSM buildings
      if (!map.current.getSource('geojson-filter')) {
        map.current.addSource('geojson-filter', {
          type: 'geojson',
          data: processedGeoJsonData
        });
      } else {
        map.current.getSource('geojson-filter').setData(processedGeoJsonData);
      }

      // Create a more sophisticated filtering approach
      // We'll use coordinate-based filtering since spatial operators might not work with composite sources
      const filterBounds = getGeojsonBounds(processedGeoJsonData);
      const [[minLon, minLat], [maxLon, maxLat]] = filterBounds;
      
      // Add some padding to ensure we catch buildings that might overlap
      const padding = 0.0002; // roughly 20 meters
      const excludeMinLon = minLon - padding;
      const excludeMaxLon = maxLon + padding;
      const excludeMinLat = minLat - padding;
      const excludeMaxLat = maxLat + padding;

      // Since complex filtering might not work with composite sources,
      // let's try a runtime approach using map events
      if (map.current.getLayer('3d-buildings')) {
        // First, restore normal visibility
        map.current.setLayoutProperty('3d-buildings', 'visibility', 'visible');
        map.current.setFilter('3d-buildings', ['==', 'extrude', 'true']);
        
        // Apply runtime filtering using queryRenderedFeatures with a delay to ensure map is rendered
        setTimeout(() => {
          isFilteringActive.current = true;
          hideBuildingsInGeojsonArea(processedGeoJsonData);
        }, 500); // Increased delay to ensure map is fully rendered
        
        // Also apply filtering when map moves to catch buildings that weren't initially visible
        const filterHandler = () => {
          if (isFilteringActive.current) {
            hideBuildingsInGeojsonArea(processedGeoJsonData);
          }
        };
        
        map.current.on('idle', filterHandler);
        
        // Store the handler for cleanup
        map.current._geojsonFilterHandler = filterHandler;
      }
      
      // Decide whether to show 2D or 3D based on mapState
      const isConfigurationMode = mapState === 'configuration';
      console.log(`🎨 Creating GeoJSON layers with visibility: ${shouldShowGeojsonLayers ? 'visible' : 'none'}`);

      if (isConfigurationMode) {
        // In configuration mode, show 2D polygons for easy clicking
        safelyAddLayer({
          id: fillLayerId,
          type: 'fill',
          source: sourceId,
          layout: {
            visibility: shouldShowGeojsonLayers ? 'visible' : 'none'
          },
          paint: {
            'fill-color': [
              'case',
              ['boolean', ['feature-state', 'selected'], false],
              '#8A2BE2', // Purple when selected
              '#FFC107'  // Yellow when not selected
            ],
            'fill-opacity': [
              'case',
              ['boolean', ['feature-state', 'selected'], false],
              0.8,  // More opaque when selected
              0.5   // Less opaque when not selected
            ]
          }
        }, null, activeGeojsonLayerId, 'geojson'); // Register with layer ID
        
        // Add polygon click handler for configuration
        map.current.off('click', fillLayerId); // Remove any existing handlers
        map.current.on('click', fillLayerId, (e) => {
          // Prevent event from bubbling to the map
          if (e.originalEvent) {
            e.originalEvent.preventDefault();
            e.originalEvent.stopPropagation();
          }
          
          if (e.features && e.features.length > 0) {
            const feature = e.features[0];
            const featureIndex = feature.id;
            
            console.log('Polygon clicked:', feature, 'Index:', featureIndex);
            
            // Immediately update the visual selection for better responsiveness
            if (featureIndex !== undefined) {
              // Clear all existing selections
              if (processedGeoJsonData.features) {
                processedGeoJsonData.features.forEach((f, index) => {
                  try {
                    map.current.setFeatureState({
                      source: sourceId,
                      id: index
                    }, {
                      selected: false
                    });
                  } catch (error) {
                    // Ignore errors
                  }
                });
              }
              
              // Set the clicked feature as selected
              try {
                map.current.setFeatureState({
                  source: sourceId,
                  id: featureIndex
                }, {
                  selected: true
                });
              } catch (error) {
                console.error('Error setting feature state:', error);
              }
            }
            
            // Pass the feature with index to the parent component
            const featureWithIndex = {
              ...feature,
              featureIndex: featureIndex
            };
            onPolygonClick && onPolygonClick(featureWithIndex);
          }
        });
        
        // Change cursor on hover
        map.current.on('mouseenter', fillLayerId, () => {
          map.current.getCanvas().style.cursor = 'pointer';
        });
        
        map.current.on('mouseleave', fillLayerId, () => {
          map.current.getCanvas().style.cursor = '';
        });
      } else {
        // In normal mode, show as flat 2D layer (no extrusion)
        // The 3D extrusion should only appear in the independent LoD1 layer
        safelyAddLayer({
          id: fillLayerId,
          type: 'fill',
          source: sourceId,
          layout: {
            visibility: shouldShowGeojsonLayers ? 'visible' : 'none'
          },
          paint: {
            'fill-color': '#FFC107',  // Yellow for normal view
            'fill-opacity': 0.5
          }
        }, null, activeGeojsonLayerId, 'geojson'); // Register with layer ID
      }
      
      // Add stroke/outline layer for better visibility
      safelyAddLayer({
        id: strokeLayerId,
        type: 'line',
        source: sourceId,
        layout: {
          visibility: shouldShowGeojsonLayers ? 'visible' : 'none'
        },
        paint: {
          'line-color': [
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            '#8A2BE2', // Purple when selected
            '#808080'  // Grey when not selected
          ],
          'line-width': [
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            4,  // Thicker when selected
            3   // Normal thickness when not selected
          ],
          'line-opacity': 1
        }
      }, null, activeGeojsonLayerId, 'geojson'); // Register with layer ID

      // Add click handlers for GeoJSON polygons (both extrusion and stroke layers)
      const handleGeojsonClick = (e) => {
        if (e.features.length > 0) {
          const feature = e.features[0];
          console.log('GeoJSON polygon clicked:', feature);
          onPolygonClick && onPolygonClick(feature);
        }
      };

      // Add click events for both layers
      map.current.on('click', extrusionLayerId, handleGeojsonClick);
      map.current.on('click', strokeLayerId, handleGeojsonClick);
      
      // Change cursor on hover for both layers
      map.current.on('mouseenter', extrusionLayerId, () => {
        map.current.getCanvas().style.cursor = 'pointer';
      });
      map.current.on('mouseleave', extrusionLayerId, () => {
        map.current.getCanvas().style.cursor = '';
      });
      map.current.on('mouseenter', strokeLayerId, () => {
        map.current.getCanvas().style.cursor = 'pointer';
      });
      map.current.on('mouseleave', strokeLayerId, () => {
        map.current.getCanvas().style.cursor = '';
      });

      // Store event handlers for cleanup
      map.current._geojsonClickHandler = handleGeojsonClick;

      // // Debug: log coordinates and height information
      // console.log('GeoJSON bounds calculation:');
      // processedGeoJsonData.features.forEach((f, i) => {
      //   console.log(`Feature ${i} coords:`, f.geometry.coordinates);
      //   console.log(`Feature ${i} height:`, f.properties.height);
      //   console.log(`Feature ${i} base_height:`, f.properties.base_height);
      // });
      
      // Calculate bounds to check if coordinates are in expected range
      const bounds = getGeojsonBounds(processedGeoJsonData);
      console.log('Calculated GeoJSON bounds:', bounds);
      console.log('Expected Tokyo area bounds should be around:', [[139.6, 35.6], [139.8, 35.8]]);
    } else {
      console.log('Hiding GeoJSON, cleaning up GeoJSON-specific resources...');
      
      // Clean up GeoJSON event listeners (building restoration is handled by unified logic)
      if (map.current._geojsonFilterHandler) {
        map.current.off('idle', map.current._geojsonFilterHandler);
        delete map.current._geojsonFilterHandler;
      }
      
      // Remove the filter source
      if (map.current.getSource('geojson-filter')) {
        map.current.removeSource('geojson-filter');
      }
      
      // Remove polygon click handlers
      if (map.current.getLayer(fillLayerId)) {
        map.current.off('click', fillLayerId);
        map.current.off('mouseenter', fillLayerId);
        map.current.off('mouseleave', fillLayerId);
      }

      // Note: Building restoration is now handled by the unified building hiding logic
    }
    };  // End of renderGeoJsonLayers function

    // If style is not loaded, wait for it to load and then retry
    if (!map.current.isStyleLoaded()) {
      console.log('⏸️ GeoJSON useEffect: Map style not loaded, waiting for styledata event...');
      const onStyleData = () => {
        console.log('✅ Map style loaded, retrying GeoJSON rendering');
        // Remove the listener first
        map.current.off('styledata', onStyleData);
        // Force a re-render by calling the rendering logic directly
        renderGeoJsonLayers();
      };
      map.current.once('styledata', onStyleData);
      return;
    }

    // Call the rendering function immediately if style is loaded
    renderGeoJsonLayers();

    // Clean up on unmount or prop change
    return () => {
      // Only remove layers if GeoJSON shouldn't be shown anymore or if we're unmounting
      if (!map.current) return;

      // Check if layers should remain (in configuration mode or showGeojson is true)
      // If so, don't remove them to preserve feature state
      const shouldKeepLayers = (showGeojson || mapState === 'configuration') && geojsonData;

      if (shouldKeepLayers) {
        console.log('🔒 Cleanup: Keeping GeoJSON layers to preserve state');
        return; // Don't remove layers
      }

      console.log('🧹 Cleanup: Removing GeoJSON layers');

      // Define layer and source IDs in cleanup scope
      const sourceId = 'geojson-mask';
      const extrusionLayerId = 'geojson-mask-extrusion';
      const fillLayerId = 'geojson-mask-fill';
      const strokeLayerId = 'geojson-mask-stroke';

      const layersToRemove = [extrusionLayerId, fillLayerId, strokeLayerId];
      layersToRemove.forEach(layer => {
        if (map.current && map.current.getLayer(layer)) {
          map.current.removeLayer(layer);
        }
      });
      
      // Remove polygon click handlers
      if (map.current && map.current.getLayer(fillLayerId)) {
        map.current.off('click', fillLayerId);
        map.current.off('mouseenter', fillLayerId);
        map.current.off('mouseleave', fillLayerId);
      }
      
      const sourcesToRemove = [sourceId, 'geojson-filter'];
      sourcesToRemove.forEach(source => {
        if (map.current && map.current.getSource(source)) {
          map.current.removeSource(source);
        }
      });
      
      // Remove event listener before clearing filters
      if (map.current && map.current._geojsonFilterHandler) {
        map.current.off('idle', map.current._geojsonFilterHandler);
        delete map.current._geojsonFilterHandler;
      }
      
      // Remove LoD1 event listener if it exists
      if (map.current && map.current._lod1FilterHandler) {
        map.current.off('idle', map.current._lod1FilterHandler);
        delete map.current._lod1FilterHandler;
      }
      
      // Remove GeoJSON click event handlers
      if (map.current && map.current._geojsonClickHandler) {
        map.current.off('click', extrusionLayerId, map.current._geojsonClickHandler);
        map.current.off('click', strokeLayerId, map.current._geojsonClickHandler);
        delete map.current._geojsonClickHandler;
      }


      // Note: Building restoration is handled by the unified building hiding logic
    };
  }, [showGeojson, geojsonData, mapState, lod1Layers, visibleLayers, organizedLayers]);

  // Handle selected polygon highlighting in configuration mode
  useEffect(() => {
    if (!map.current || mapState !== 'configuration') return;

    const updateSelection = () => {
      const sourceId = 'geojson-mask';
      const source = map.current.getSource(sourceId);

      if (!source || !(showGeojson || mapState === 'configuration') || !geojsonData) {
        console.log('🔮 Selection update skipped: source or data not ready');
        return;
      }

      console.log('🔮 Updating polygon selection state:', selectedBuilding);
    
    // Clear all existing selections first
    if (geojsonData.features) {
      geojsonData.features.forEach((feature, index) => {
        try {
          map.current.setFeatureState({
            source: sourceId,
            id: index
          }, {
            selected: false
          });
        } catch (error) {
          // Ignore errors for features that might not exist
        }
      });
    }

    // Set the selected feature if there is one
    if (selectedBuilding && selectedBuilding.featureIndex !== undefined) {
      try {
        console.log(`🔮 Setting selected state for feature index: ${selectedBuilding.featureIndex}`);
        map.current.setFeatureState({
          source: sourceId,
          id: selectedBuilding.featureIndex
        }, {
          selected: true
        });
      } catch (error) {
        console.error('Error setting feature state:', error);
      }
    }
    
    // Ensure building colors remain consistent after polygon selection
    if (geojsonData && map.current.getLayer('3d-buildings')) {
      try {
        // Only respect the height color toggle, no configuration mode logic
        if (!showHeightColors) {
          // Force white buildings when height colors are disabled
          map.current.setPaintProperty('3d-buildings', 'fill-extrusion-color', [
            'case',
            ['boolean', ['feature-state', 'hidden'], false],
            'rgba(0,0,0,0)', // Completely transparent if hidden
            '#ffffff' // White for visible buildings
          ]);
          map.current.setPaintProperty('3d-buildings', 'fill-extrusion-opacity', 0.8);
        }
        // If height colors are enabled, the main useEffect will handle the coloring
      } catch (error) {
        console.log('Building layer not ready for color update');
      }
    }
    };  // End of updateSelection function

    // Wait for style to load if necessary, then call updateSelection
    if (!map.current.isStyleLoaded()) {
      console.log('🔮 Style not loaded, waiting for styledata event...');
      const onStyleData = () => {
        console.log('🔮 Style loaded, updating selection');
        map.current.off('styledata', onStyleData);
        updateSelection();
      };
      map.current.once('styledata', onStyleData);
    } else {
      // Style is already loaded, call immediately
      updateSelection();
    }
  }, [selectedBuilding, showHeightColors, mapState, showGeojson, geojsonData]); // Added back necessary dependencies

  // Handle LoD1 layers visualization - Simplified to follow GeoJSON display logic
  useEffect(() => {
    if (!map.current) {
      return;
    }

    // If style is not loaded, wait for it to load
    if (!map.current.isStyleLoaded()) {
      const handleStyleLoad = () => {
        processLod1Layers();
      };
      map.current.once('styledata', handleStyleLoad);
      return () => {
        map.current?.off('styledata', handleStyleLoad);
      };
    }

    processLod1Layers();

    function processLod1Layers() {
      if (!map.current) return;

    // Clean up previous LoD1 event handler
    if (map.current && map.current._lod1FilterHandler) {
      map.current.off('idle', map.current._lod1FilterHandler);
      delete map.current._lod1FilterHandler;
    }

    // Clean up all existing LoD1 layers first
    const mapLayers = map.current.getStyle().layers || [];
    mapLayers.forEach(layer => {
      if (layer.id.includes('-extrusion') && layer.id.startsWith('lod1-')) {
        try {
          map.current.removeLayer(layer.id);
          unregisterLayer(layer.id);
          const sourceId = layer.id.replace('-extrusion', '-source');
          if (map.current.getSource(sourceId)) {
            map.current.removeSource(sourceId);
          }
        } catch (err) {
          console.error(`Error removing layer ${layer.id}:`, err);
        }
      }
    });

    // Add LoD1 layers if they exist and should be visible
    if (lod1Layers && lod1Layers.length > 0) {
      let anyLod1LayerVisible = false;
      let combinedLod1Data = { type: 'FeatureCollection', features: [] };

      lod1Layers.forEach(layer => {
        const shouldShow = visibleLayers.has(layer.id);

        if (shouldShow) {
          anyLod1LayerVisible = true;
          const sourceId = `${layer.id}-source`;
          const layerId = `${layer.id}-extrusion`;
          const controlId = layer.controlId || layer.id; // Use top-level controlId, not metadata

          // Process the LoD1 data to ensure proper height properties
          const processedData = processGeojsonHeights(layer.data);
          
          // Combine all LoD1 data for building filtering
          if (processedData.features) {
            combinedLod1Data.features.push(...processedData.features);
          }

          // Add source
          map.current.addSource(sourceId, {
            type: 'geojson',
            data: processedData
          });

          // Add 3D extrusion layer
          safelyAddLayer({
            id: layerId,
            type: 'fill-extrusion',
            source: sourceId,
            layout: {
              visibility: 'visible'
            },
            paint: {
              'fill-extrusion-color': '#FFC107', // Yellow color for LoD1 models
              'fill-extrusion-height': ['get', 'height'],
              'fill-extrusion-base': ['get', 'base_height'],
              'fill-extrusion-opacity': 0.8
            }
          }, null, controlId, 'lod1');
        }
      });
      
      // Collect bounding boxes from visible LoD2 layers (from orthophoto uploads)
      const orthophotoBBoxes = [];
      if (organizedLayers && organizedLayers.lod2) {
        organizedLayers.lod2.forEach(layer => {
          if (visibleLayers.has(layer.id)) {
            // Check both data.bbox and metadata.bbox
            const bbox = layer.data?.bbox || layer.metadata?.bbox;
            if (bbox) {
              orthophotoBBoxes.push(bbox);
              console.log(`   ├─ Found orthophoto bbox for layer ${layer.id}:`, bbox);
            }
          }
        });
      }

      // Hide overlapping buildings for LoD1 layers and/or orthophoto bboxes
      const hasLod1Data = anyLod1LayerVisible && combinedLod1Data.features.length > 0;
      const hasOrthophotoBBoxes = orthophotoBBoxes.length > 0;

      if (hasLod1Data || hasOrthophotoBBoxes) {
        if (hasLod1Data) {
          console.log(`🏗️ LoD1 layers visible, hiding overlapping buildings`);
          console.log(`   ├─ Combined LoD1 features: ${combinedLod1Data.features.length}`);
        }
        if (hasOrthophotoBBoxes) {
          console.log(`📍 Orthophoto bboxes found, hiding buildings in ${orthophotoBBoxes.length} bbox(es)`);
        }
        console.log(`   ├─ Current filtering active: ${isFilteringActive.current}`);

        // Use setTimeout to ensure layers are fully rendered before filtering
        setTimeout(() => {
          hideBuildingsInGeojsonArea(hasLod1Data ? combinedLod1Data : null, orthophotoBBoxes);
          isFilteringActive.current = true;
          console.log(`   └─ Building hiding applied`);
        }, 500);

        // Also apply filtering when map moves to catch buildings that weren't initially visible
        const filterHandler = () => {
          if (isFilteringActive.current) {
            // Re-collect bboxes in case visibility changed
            const bboxes = [];
            if (organizedLayers && organizedLayers.lod2) {
              organizedLayers.lod2.forEach(layer => {
                if (visibleLayers.has(layer.id)) {
                  // Check both data.bbox and metadata.bbox
                  const bbox = layer.data?.bbox || layer.metadata?.bbox;
                  if (bbox) {
                    bboxes.push(bbox);
                  }
                }
              });
            }
            hideBuildingsInGeojsonArea(hasLod1Data ? combinedLod1Data : null, bboxes);
          }
        };

        map.current.on('idle', filterHandler);

        // Store the handler for cleanup
        map.current._lod1FilterHandler = filterHandler;
      }
    }
    } // End processLod1Layers

    // Add LoD3 layers if they exist and should be visible
    if (lod3Layers && lod3Layers.length > 0) {
      lod3Layers.forEach(layer => {
        const shouldShow = visibleLayers.has(layer.id);
        console.log(`LoD3 layer ${layer.id} should show: ${shouldShow}`);
        
        if (shouldShow) {
          const layerId = `${layer.id}-3d`;
          console.log(`Adding LoD3 layer: ${layerId}`);

          // Create custom layer for 3D OBJ model
          const customLayer = {
            id: layerId,
            type: 'custom',
            renderingMode: '3d',
            onAdd: function(map, gl) {
              // ObjModelViewer will handle the 3D rendering
              this.map = map;
            },
            render: function(gl, matrix) {
              // Rendering is handled by ObjModelViewer component
            }
          };

          // Add the custom layer to the map
          if (!map.current.getLayer(layerId)) {
            map.current.addLayer(customLayer);
            console.log(`✅ Added LoD3 custom layer: ${layerId}`);
          }

          // Register the layer for tracking
          registerLayer(layerId, layer.id, 'lod3');
        } else {
          // Remove layer if it should not be visible
          const layerId = `${layer.id}-3d`;
          if (map.current.getLayer(layerId)) {
            map.current.removeLayer(layerId);
            console.log(`❌ Removed LoD3 layer: ${layerId}`);
          }
          unregisterLayer(layerId);
        }
      });
    }

  }, [lod1Layers, lod3Layers, visibleLayers, anyObjModelVisible]);

  // Function to update building visibility based on hidden state
  function updateBuildingVisibility() {
    if (!map.current || !map.current.getLayer('3d-buildings')) return;

    try {
      // Set fixed opacity based on height colors toggle (no data expressions)
      map.current.setPaintProperty('3d-buildings', 'fill-extrusion-opacity', showHeightColors ? 1 : 0.8);
      
      // Use color to hide buildings (set to transparent for hidden buildings)
      if (showHeightColors) {
        map.current.setPaintProperty('3d-buildings', 'fill-extrusion-color', [
          'case',
          ['boolean', ['feature-state', 'hidden'], false],
          'rgba(0,0,0,0)', // Completely transparent if hidden
          [
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            '#8A2BE2', // Purple when selected
            [
              'interpolate',
              ['linear'],
              ['get', 'height'],
              0, '#2196F3',    // 0-10m: Blue
              10, '#4CAF50',   // 10-20m: Green
              20, '#FFEB3B',   // 20-30m: Yellow
              30, '#FF9800',   // 30-45m: Orange
              45, '#F44336'    // 45m+: Red
            ]
          ]
        ]);
      } else {
        map.current.setPaintProperty('3d-buildings', 'fill-extrusion-color', [
          'case',
          ['boolean', ['feature-state', 'hidden'], false],
          'rgba(0,0,0,0)', // Completely transparent if hidden
          [
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            '#8A2BE2', // Purple when selected
            '#ffffff'  // White for normal buildings
          ]
        ]);
      }
    } catch (error) {
      console.error('Error updating building visibility:', error);
    }
  }

  // Function to hide buildings within the GeoJSON area and orthophoto bounding boxes
  function hideBuildingsInGeojsonArea(geojsonData, orthophotoBBoxes = []) {
    if (!map.current || !map.current.isStyleLoaded()) {
      console.log('Map not ready for filtering');
      return;
    }

    try {
      // Query all visible buildings on the current view
      const allBuildings = map.current.queryRenderedFeatures({
        layers: ['3d-buildings']
      });

      let hiddenCount = 0;
      let checkedCount = 0;

      // Check each building to see if it overlaps with GeoJSON polygons or orthophoto bboxes
      allBuildings.forEach(building => {
        if (building.id) {
          checkedCount++;

          // Check if building is in GeoJSON area
          const inGeojson = geojsonData && isBuildingInGeojsonArea(building, geojsonData);

          // Check if building is in any orthophoto bounding box
          const inOrthophotoBBox = orthophotoBBoxes.length > 0 &&
                                   orthophotoBBoxes.some(bbox => isBuildingInBBox(building, bbox));

          // Hide if in either GeoJSON area or orthophoto bbox
          if (inGeojson || inOrthophotoBBox) {
            // Only hide if not already hidden
            if (!hiddenBuildingIds.current.has(building.id)) {
              hiddenBuildingIds.current.add(building.id);

              try {
                // Set feature state to hide the building
                map.current.setFeatureState({
                  source: 'composite',
                  sourceLayer: 'building',
                  id: building.id
                }, {
                  hidden: true
                });
                hiddenCount++;
              } catch (error) {
                console.warn('Failed to hide building:', building.id, error);
              }
            }
          }
        }
      });

      // Update building paint properties to respect the hidden state
      updateBuildingVisibility();

    } catch (error) {
      console.error('Error filtering buildings:', error);
    }
  }

  // Function to check if a building intersects with GeoJSON area
  function isBuildingInGeojsonArea(buildingFeature, geojsonData) {
    if (!buildingFeature.geometry || !buildingFeature.geometry.coordinates) {
      return false;
    }
    
    try {
      // Get building geometry
      let buildingCoords;
      if (buildingFeature.geometry.type === 'Polygon') {
        buildingCoords = buildingFeature.geometry.coordinates[0];
      } else if (buildingFeature.geometry.type === 'MultiPolygon') {
        // For MultiPolygon, use the first polygon
        buildingCoords = buildingFeature.geometry.coordinates[0][0];
      } else {
        return false;
      }
      
      // Calculate building center for initial quick check
      const sumX = buildingCoords.reduce((sum, coord) => sum + coord[0], 0);
      const sumY = buildingCoords.reduce((sum, coord) => sum + coord[1], 0);
      const buildingCenter = [sumX / buildingCoords.length, sumY / buildingCoords.length];
      
      // Check if building center is within any GeoJSON polygon (quick check)
      let centerInside = false;
      for (const feature of geojsonData.features) {
        if (feature.geometry.type === 'Polygon') {
          if (isPointInPolygon(buildingCenter, feature.geometry.coordinates[0])) {
            centerInside = true;
            break;
          }
        } else if (feature.geometry.type === 'MultiPolygon') {
          for (const polygon of feature.geometry.coordinates) {
            if (isPointInPolygon(buildingCenter, polygon[0])) {
              centerInside = true;
              break;
            }
          }
          if (centerInside) break;
        }
      }
      
      // If center is inside, definitely overlapping
      if (centerInside) {
        return true;
      }
      
      // Check if any building vertex is within any GeoJSON polygon
      for (const point of buildingCoords) {
        for (const feature of geojsonData.features) {
          if (feature.geometry.type === 'Polygon') {
            if (isPointInPolygon(point, feature.geometry.coordinates[0])) {
              return true;
            }
          } else if (feature.geometry.type === 'MultiPolygon') {
            for (const polygon of feature.geometry.coordinates) {
              if (isPointInPolygon(point, polygon[0])) {
                return true;
              }
            }
          }
        }
      }
      
      // Check if any GeoJSON vertex is within the building polygon
      for (const feature of geojsonData.features) {
        if (feature.geometry.type === 'Polygon') {
          for (const point of feature.geometry.coordinates[0]) {
            if (isPointInPolygon(point, buildingCoords)) {
              return true;
            }
          }
        } else if (feature.geometry.type === 'MultiPolygon') {
          for (const polygon of feature.geometry.coordinates) {
            for (const point of polygon[0]) {
              if (isPointInPolygon(point, buildingCoords)) {
                return true;
              }
            }
          }
        }
      }
      
      return false;
    } catch (error) {
      console.warn('Error checking building overlap:', error);
      return false;
    }
  }

  // Simple point-in-polygon test using ray casting algorithm
  function isPointInPolygon(point, polygon) {
    const [x, y] = point;
    let inside = false;

    for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
      const [xi, yi] = polygon[i];
      const [xj, yj] = polygon[j];

      if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
        inside = !inside;
      }
    }

    return inside;
  }

  // Function to check if a building is within a bounding box
  function isBuildingInBBox(buildingFeature, bbox) {
    if (!buildingFeature.geometry || !buildingFeature.geometry.coordinates || !bbox) {
      return false;
    }

    try {
      // Get building geometry
      let buildingCoords;
      if (buildingFeature.geometry.type === 'Polygon') {
        buildingCoords = buildingFeature.geometry.coordinates[0];
      } else if (buildingFeature.geometry.type === 'MultiPolygon') {
        buildingCoords = buildingFeature.geometry.coordinates[0][0];
      } else {
        return false;
      }

      // Calculate building center
      const sumX = buildingCoords.reduce((sum, coord) => sum + coord[0], 0);
      const sumY = buildingCoords.reduce((sum, coord) => sum + coord[1], 0);
      const centerX = sumX / buildingCoords.length;
      const centerY = sumY / buildingCoords.length;

      // Check if center is within bounding box
      return centerX >= bbox.west && centerX <= bbox.east &&
             centerY >= bbox.south && centerY <= bbox.north;
    } catch (error) {
      console.warn('Error checking building bbox overlap:', error);
      return false;
    }
  }

  // Function to clear building filters
  function clearBuildingFilters() {
    if (!map.current) return;
    
    try {
      // Restore visibility for all hidden buildings
      hiddenBuildingIds.current.forEach(buildingId => {
        try {
          map.current.setFeatureState({
            source: 'composite',
            sourceLayer: 'building',
            id: buildingId
          }, {
            hidden: false
          });
        } catch (error) {
          console.warn('Error restoring building:', buildingId, error);
        }
      });

      // Clear the hidden building IDs
      hiddenBuildingIds.current.clear();

      // Reset building paint properties to normal state
      updateBuildingVisibility();
      
    } catch (error) {
      console.error('Error clearing building filters:', error);
    }
  }

  // Function to load and display OBJ model using ObjModelViewer
  async function loadAndDisplayObjModel() {
    console.log('🎬 Activating OBJ model display...');
    setShowObjModel(true);
  }

  // Placeholder for future OBJ helper functions - to be implemented cleanly
  // These will be replaced with proper ObjModelViewer integration

  // Function to remove OBJ model
  const removeObjModel = () => {
    console.log('🧹 Deactivating OBJ model display...');
    setShowObjModel(false);
  };

  // Simple building creation helper - placeholder for clean implementation
  const createBuildingFromFaces = (vertices, faces) => {
    console.log('Building from faces creation - to be implemented cleanly');
    return [];
  };

  // Simple building footprint helper - placeholder for clean implementation
  const createSimpleBuildingFootprint = (vertices) => {
    console.log('Simple building footprint - to be implemented cleanly');
    return [];
  };

  // Utility to calculate bounds of geojson
  function getGeojsonBounds(geojson) {
    let minLon = Infinity, minLat = Infinity, maxLon = -Infinity, maxLat = -Infinity;
    
    function processCoords(coords) {
      if (typeof coords[0] === 'number') {
        const [lon, lat] = coords;
        minLon = Math.min(minLon, lon);
        maxLon = Math.max(maxLon, lon);
        minLat = Math.min(minLat, lat);
        maxLat = Math.max(maxLat, lat);
      } else {
        coords.forEach(processCoords);
      }
    }
    
    // Process all features
    geojson.features.forEach(f => {
      if (f.geometry && f.geometry.coordinates) {
        processCoords(f.geometry.coordinates);
      }
    });
    
    console.log('Calculated bounds:', { minLon, minLat, maxLon, maxLat });
    return [[minLon, minLat], [maxLon, maxLat]];
  }

  // Expose zoomToGeojson and zoomToOrthophoto to parent
  useImperativeHandle(ref, () => ({
    zoomToGeojson: (geojsonDataParam) => {
      const dataToUse = geojsonDataParam || geojsonData;
      if (dataToUse && map.current) {
        const bounds = getGeojsonBounds(dataToUse);
        console.log('Zooming to bounds:', bounds);
        
        // Validate bounds
        if (bounds[0][0] !== Infinity && bounds[0][1] !== Infinity && 
            bounds[1][0] !== -Infinity && bounds[1][1] !== -Infinity) {
          map.current.fitBounds(bounds, { 
            padding: 50,
            maxZoom: 18,
            duration: 2000
          });

          // Calculate and display center with validation
          const centerLng = (bounds[0][0] + bounds[1][0]) / 2;
          const centerLat = (bounds[0][1] + bounds[1][1]) / 2;
          
          // Validate center coordinates
          if (typeof centerLng === 'number' && typeof centerLat === 'number' && 
              !isNaN(centerLng) && !isNaN(centerLat) && 
              centerLng >= -180 && centerLng <= 180 && 
              centerLat >= -90 && centerLat <= 90) {
            const center = [centerLng, centerLat];
            console.log('✅ Valid GeoJSON center:', center);
            
            // Store the calculated center for OBJ model alignment
            setGeojsonCenter(center);
            
            // Remove any existing marker first
            // const existingMarkers = document.querySelectorAll('.mapboxgl-marker');
            // existingMarkers.forEach(marker => marker.remove());
            
            // Add marker at center
            // new mapboxgl.Marker({ color: '#ff0000' })
            //   .setLngLat(center)
            //   .addTo(map.current);
          } else {
            console.error('❌ Invalid GeoJSON center coordinates:', [centerLng, centerLat]);
          }
        } else {
          console.error('Invalid bounds calculated:', bounds);
        }
      }
    },
    zoomToOrthophoto: (layerId) => {
      if (map.current && layerId) {
        const source = map.current.getSource(layerId);
        if (source && source.type === 'image' && source.coordinates) {
          // coordinates: [top-left, top-right, bottom-right, bottom-left]
          const coords = source.coordinates;
          const lons = coords.map(c => c[0]);
          const lats = coords.map(c => c[1]);
          const minLon = Math.min(...lons);
          const maxLon = Math.max(...lons);
          const minLat = Math.min(...lats);
          const maxLat = Math.max(...lats);
          const bounds = [[minLon, minLat], [maxLon, maxLat]];
          map.current.fitBounds(bounds, { padding: 40 });
          const center = [
            (minLon + maxLon) / 2,
            (minLat + maxLat) / 2
          ];
          //new mapboxgl.Marker().setLngLat(center).addTo(map.current);
        }
      }
    },
    zoomToLod3: (layerId) => {
      console.log('🎯 Zooming to LoD3 model:', layerId);
      
      // LoD3 models are positioned at the GeoJSON center, so zoom to that
      if (geojsonData && map.current) {
        const bounds = getGeojsonBounds(geojsonData);
        console.log('Zooming to LoD3 model bounds:', bounds);
        
        // Validate bounds
        if (bounds[0][0] !== Infinity && bounds[0][1] !== Infinity && 
            bounds[1][0] !== -Infinity && bounds[1][1] !== -Infinity) {
          // Use a closer zoom level for LoD3 models to see more detail
          map.current.fitBounds(bounds, { 
            padding: 30,  // Less padding for closer view
            maxZoom: 20,  // Higher max zoom for detail
            duration: 1500
          });

          // Calculate and mark the center where the LoD3 model is positioned
          const center = [
            (bounds[0][0] + bounds[1][0]) / 2,
            (bounds[0][1] + bounds[1][1]) / 2
          ];
          console.log('LoD3 model center:', center);
          
          // Remove any existing marker first
          const existingMarkers = document.querySelectorAll('.mapboxgl-marker');
          existingMarkers.forEach(marker => marker.remove());
          
          // Add marker at LoD3 model center with different color
          // new mapboxgl.Marker({ color: '#00ff00' })  // Green marker for LoD3
          //   .setLngLat(center)
          //   .addTo(map.current);
        } else {
          console.error('Invalid bounds for LoD3 zoom:', bounds);
        }
      } else {
        console.warn('No GeoJSON data available for LoD3 zoom');
      }
    },
    zoomToObjModel: async (objPath, layerId = null) => {
      console.log('🎯 Zooming to OBJ model coordinates:', objPath);
      
      if (!map.current) {
        console.warn('Map not available for OBJ zoom');
        return;
      }
      
      try {
        // Import the coordinate utilities
        const { loadAndAnalyzeObjCoordinates } = await import('../utils/coordinateUtils.js');
        
        // Load and analyze OBJ coordinates
        const coordinateAnalysis = await loadAndAnalyzeObjCoordinates(objPath);
        const centerWGS84 = coordinateAnalysis.centerWGS84; // [lng, lat]
        const dimensions = coordinateAnalysis.dimensions;
        
        console.log('🎯 OBJ model center (WGS84):', centerWGS84);
        console.log('🎯 OBJ model dimensions:', dimensions);
        
        // Calculate appropriate zoom level based on model size
        const maxDimension = Math.max(dimensions.width, dimensions.depth);
        let zoomLevel;
        if (maxDimension > 500) {
          zoomLevel = 16;
        } else if (maxDimension > 200) {
          zoomLevel = 17;
        } else if (maxDimension > 100) {
          zoomLevel = 18;
        } else {
          zoomLevel = 19;
        }
        
        // Zoom to the OBJ model center with appropriate level
        map.current.flyTo({
          center: centerWGS84,
          zoom: zoomLevel,
          pitch: 60, // Good angle for viewing 3D models
          bearing: 0,
          duration: 2000
        });
        
        // Remove any existing marker first
        // const existingMarkers = document.querySelectorAll('.mapboxgl-marker');
        // existingMarkers.forEach(marker => marker.remove());
        
        // // Add marker at OBJ model center
        // const markerColor = layerId && layerId.includes('lod3') ? '#00ff00' : '#ff6600'; // Green for LoD3, Orange for LoD2
        // new mapboxgl.Marker({ color: markerColor })
        //   .setLngLat(centerWGS84)
        //   .addTo(map.current);
        
        console.log(`✅ Successfully zoomed to OBJ model at: [${centerWGS84[0].toFixed(6)}, ${centerWGS84[1].toFixed(6)}]`);
        
      } catch (error) {
        console.error('❌ Failed to zoom to OBJ model:', error);
        // Fallback to GeoJSON zoom if OBJ coordinate analysis fails
        console.log('🔄 Falling back to GeoJSON zoom');
        if (geojsonData) {
          const bounds = getGeojsonBounds(geojsonData);
          map.current.fitBounds(bounds, { padding: 40 });
        }
      }
    },
    registerExternalLayer: (layerId, controlId) => {
      return registerExternalLayer(layerId, controlId);
    }
  }), [geojsonData]);

  // Optionally, trigger zoom when zoomGeojsonTrigger changes
  useEffect(() => {
    if (zoomGeojsonTrigger && geojsonData && map.current) {
      const bounds = getGeojsonBounds(geojsonData);
      map.current.fitBounds(bounds, { padding: 40 });
    }
  }, [zoomGeojsonTrigger, geojsonData]);

  // Calculate and store center when geojsonData changes
  useEffect(() => {
    if (geojsonData && !geojsonCenter) {
      const bounds = getGeojsonBounds(geojsonData);
      if (bounds[0][0] !== Infinity && bounds[0][1] !== Infinity && 
          bounds[1][0] !== -Infinity && bounds[1][1] !== -Infinity) {
        const centerLng = (bounds[0][0] + bounds[1][0]) / 2;
        const centerLat = (bounds[0][1] + bounds[1][1]) / 2;
        
        // Validate center coordinates
        if (typeof centerLng === 'number' && typeof centerLat === 'number' && 
            !isNaN(centerLng) && !isNaN(centerLat) && 
            centerLng >= -180 && centerLng <= 180 && 
            centerLat >= -90 && centerLat <= 90) {
          const center = [centerLng, centerLat];
          console.log('📍 Auto-calculated valid GeoJSON center for OBJ alignment:', center);
          setGeojsonCenter(center);
        } else {
          console.error('❌ Invalid auto-calculated center coordinates:', [centerLng, centerLat]);
          // Set a fallback center for Tokyo area
          setGeojsonCenter([139.7, 35.7]);
        }
      }
    }
  }, [geojsonData, geojsonCenter]);

  return (
    <>
      <div ref={mapContainer} className="map-container" />
      
      {/* OBJ Model Viewer */}
      {showObjModel && map.current && (
        <ObjModelViewer
          map={map.current}
          objPath="/src/assets/test_data/01/result_lod2.obj"
          showBoundingBox={false} // Enable for debugging
          onModelLoaded={(data) => {
            console.log('🎭 LoD2 OBJ Model loaded successfully:', data);
          }}
        />
      )}

      {/* LoD1 OBJ Model Viewers (for city assets and other OBJ-based LoD1 layers) */}
      {lod1Layers && map.current && lod1Layers.map(layer => {
        const shouldShow = visibleLayers.has(layer.id);
        // Only render if this LoD1 layer has objPath (backend-generated OBJ models like city assets)
        const hasObjPath = layer.data && layer.data.objPath;
        if (!shouldShow || !hasObjPath) return null;

        return (
          <ObjModelViewer
            key={layer.id}
            map={map.current}
            objPath={layer.data.url || layer.data.objPath}
            mtlPath={layer.data.mtlPath}
            showBoundingBox={false} // Enable for debugging
            onModelLoaded={(data) => {
              console.log(`🎭 LoD1 OBJ Model ${layer.id} loaded successfully:`, data);
            }}
          />
        );
      })}

      {/* LoD3 OBJ Model Viewers */}
      {lod3Layers && map.current && lod3Layers.map(layer => {
        const shouldShow = visibleLayers.has(layer.id);
        if (!shouldShow) return null;

        return (
          <ObjModelViewer
            key={layer.id}
            map={map.current}
            objPath={layer.data.objPath}
            mtlPath={layer.data.mtlPath}
            showBoundingBox={false} // Enable for debugging
            onModelLoaded={(data) => {
              console.log(`🎭 LoD3 Model ${layer.id} loaded successfully:`, data);
            }}
          />
        );
      })}

      {/* Map state overlays */}
      {mapState === 'loading' && <div className="map-overlay-loading" />}
      
      {/* Loading indicator */}
      {isLoading && (
        <div className="loading-overlay">
          <div className="loading-text">Loading BridgeUI Map...</div>
        </div>
      )}
    </>
  );
});

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Compare primitive props
  if (prevProps.showGeojson !== nextProps.showGeojson) return false;
  if (prevProps.mapState !== nextProps.mapState) return false;
  if (prevProps.showHeightColors !== nextProps.showHeightColors) return false;
  if (prevProps.zoomGeojsonTrigger !== nextProps.zoomGeojsonTrigger) return false;

  // Compare object/array props by reference (for memoized props)
  if (prevProps.selectedBuilding !== nextProps.selectedBuilding) return false;
  if (prevProps.geojsonData !== nextProps.geojsonData) return false;
  if (prevProps.visibleLayers !== nextProps.visibleLayers) return false;
  if (prevProps.lod1Layers !== nextProps.lod1Layers) return false;
  if (prevProps.lod3Layers !== nextProps.lod3Layers) return false;
  if (prevProps.organizedLayers !== nextProps.organizedLayers) return false;

  // Compare callbacks (should be memoized with useCallback in parent)
  if (prevProps.onMapLoad !== nextProps.onMapLoad) return false;
  if (prevProps.onBuildingClick !== nextProps.onBuildingClick) return false;
  if (prevProps.onPolygonClick !== nextProps.onPolygonClick) return false;

  return true;
};

export default React.memo(MapContainer, arePropsEqual);