import React, { useEffect, useRef } from 'react';
import mapboxgl from 'mapbox-gl';
import * as THREE from 'three';
import { OBJLoader } from 'three/examples/jsm/loaders/OBJLoader.js';
import { MTLLoader } from 'three/examples/jsm/loaders/MTLLoader.js';
import { loadAndAnalyzeObjCoordinates } from '../utils/coordinateUtils.js';

/**
 * ObjModelViewer - A complete OBJ model viewer component for Mapbox with Three.js
 * This component handles the complete pipeline from OBJ loading to 3D visualization
 * 
 * COORDINATE POSITIONING STRATEGY:
 * Uses actual OBJ coordinates in EPSG:30169 converted to EPSG:4326 for Mapbox:
 * - Load OBJ file and extract vertex coordinates
 * - Calculate bounding box and center in EPSG:30169
 * - Convert center position to EPSG:4326 (WGS84) for Mapbox positioning
 * - Position the model at its actual geographic location
 * - No rotation or transformation, only position translation
 */
const ObjModelViewer = ({ 
  map, 
  objPath = '/src/assets/result_demo/result/r1l2/results_route1_lod2.obj',
  mtlPath = '/src/assets/result_demo/result/r1l2/material.mtl',
  altitude = 0,
  scale = 1.0,
  mirrorX = false, // Mirror along X-axis (left-right flip)
  mirrorY = false, // Mirror along Y-axis (up-down flip) 
  mirrorZ = false, // Mirror along Z-axis (front-back flip)
  onModelLoaded = null,
  showBoundingBox = false, // Show OBJ bounding box for debugging
  hideBuildingsInGeojsonArea = null, // Function to hide buildings in model area
  clearBuildingFilters = null, // Function to restore hidden buildings
  showGeojson = false, // Whether GeoJSON is currently visible to avoid duplicate hiding
  showObjModel = false, // Whether the main OBJ model is visible
  visibleLod3Layers = new Set(), // Set of visible LoD3 layer IDs
}) => {
  const customLayerRef = useRef(null);

  useEffect(() => {
    if (!map) {
      console.warn('⚠️ Map not available for OBJ model');
      return;
    }

    console.log('🎬 ObjModelViewer mounting with coordinate-based positioning...');
    console.log('   ├─ OBJ Path:', objPath);
    console.log('   ├─ MTL Path:', mtlPath);
    console.log('   ├─ Scale:', scale);
    console.log('   ├─ Altitude:', altitude);
    console.log('   ├─ Show Bounding Box:', showBoundingBox);
    console.log('   └─ Is Backend URL:', objPath?.startsWith('http'));
    
    // Additional logging for backend URLs
    if (objPath?.startsWith('http')) {
      console.log('🔗 Loading from backend URL:', objPath);
      // Test accessibility of the URL
      fetch(objPath, { method: 'HEAD' })
        .then(response => {
          console.log(`   └─ URL accessibility test: ${response.ok ? '✅ OK' : '❌ FAILED'} (${response.status})`);
        })
        .catch(error => {
          console.error('   └─ URL accessibility test failed:', error);
        });
    }
    
    let isMounted = true;
    
    const loadModelWithCoordinates = async () => {
      try {
        // Step 1: Load and analyze OBJ coordinates
        console.log('📐 Analyzing OBJ coordinates...');
        const coordinateAnalysis = await loadAndAnalyzeObjCoordinates(objPath);
        
        if (!isMounted) return;
        
        // Check if coordinate analysis was successful
        if (!coordinateAnalysis.success || !coordinateAnalysis.center30169) {
          console.error('❌ Failed to analyze OBJ coordinates, cannot render model');
          return;
        }
        
        const modelOrigin = coordinateAnalysis.centerWGS84; // [lng, lat] in WGS84
        const modelAltitude = 0; // Ground level
        
        console.log('   ├─ Using OBJ-derived position:', modelOrigin);
        console.log('   ├─ EPSG:30169 center:', [coordinateAnalysis.center30169.x, coordinateAnalysis.center30169.z]);
        console.log('   └─ Model dimensions:', coordinateAnalysis.dimensions);

        const modelAsMercatorCoordinate = mapboxgl.MercatorCoordinate.fromLngLat(
          modelOrigin,
          modelAltitude
        );

        const modelTransform = {
          translateX: modelAsMercatorCoordinate.x,
          translateY: modelAsMercatorCoordinate.y,
          translateZ: modelAsMercatorCoordinate.z,
          scale: modelAsMercatorCoordinate.meterInMercatorCoordinateUnits()
        };

        const camera = new THREE.Camera();
        const scene = new THREE.Scene();
        const renderer = new THREE.WebGLRenderer({
          canvas: map.getCanvas(),
          context: map.getCanvas().getContext('webgl'),
          antialias: true
        });

        renderer.autoClear = false;
        
        // Configure renderer for proper lighting and shadows
        console.log('🔧 Configuring renderer for lighting...');
        renderer.shadowMap.enabled = true;
        renderer.shadowMap.type = THREE.PCFSoftShadowMap; // Better shadow quality
        renderer.physicallyCorrectLights = true; // Better lighting calculations
        renderer.toneMapping = THREE.ACESFilmicToneMapping; // Better color reproduction
        renderer.toneMappingExposure = 1.0;
        renderer.outputEncoding = THREE.sRGBEncoding; // Proper color space
        console.log('   └─ Renderer configured for shadows and physically correct lighting');

        // Step 2: Load OBJ model
        console.log('📥 Loading OBJ model...');
        let object;
        
        if (mtlPath) {
          // Load with MTL materials
          console.log('🎨 Loading with MTL materials...');
          const mtlLoader = new MTLLoader();

          // Set resource path so textures can be found
          // Extract the directory path from the MTL path
          const lastSlashIndex = mtlPath.lastIndexOf('/');
          if (lastSlashIndex !== -1) {
            const resourcePath = mtlPath.substring(0, lastSlashIndex + 1);
            mtlLoader.setResourcePath(resourcePath);
            console.log(`   ├─ Resource path set to: ${resourcePath}`);
          }

          const materials = await new Promise((resolve, reject) => {
            mtlLoader.load(
              mtlPath,
              (loadedMaterials) => {
                console.log('✅ MTL file loaded successfully!');

                // Log material details
                const materialNames = Object.keys(loadedMaterials.materials);
                console.log(`   ├─ Materials found: ${materialNames.join(', ')}`);

                materialNames.forEach(name => {
                  const mat = loadedMaterials.materials[name];
                  if (mat.map) {
                    console.log(`   │  └─ ${name} has texture map`);
                  }
                });

                loadedMaterials.preload();
                resolve(loadedMaterials);
              },
              (progress) => {
                // Progress callback
                console.log(`   ├─ Loading MTL: ${progress.loaded}/${progress.total}`);
              },
              (error) => {
                console.warn('⚠️ Error loading MTL file:', error);
                resolve(null);
              }
            );
          });
          
          const objLoader = new OBJLoader();
          if (materials) {
            objLoader.setMaterials(materials);
          }
          
          object = await new Promise((resolve, reject) => {
            objLoader.load(
              objPath,
              (loadedObject) => {
                console.log('✅ OBJ file loaded successfully with materials!');
                resolve(loadedObject);
              },
              undefined,
              (error) => {
                console.error('❌ Error loading OBJ file:', error);
                reject(error);
              }
            );
          });
        } else {
          // Load without MTL materials
          console.log('📦 Loading without MTL materials...');
          const loader = new OBJLoader();
          
          object = await new Promise((resolve, reject) => {
            loader.load(
              objPath,
              (loadedObject) => {
                console.log('✅ OBJ file loaded successfully!');
                resolve(loadedObject);
              },
              undefined,
              (error) => {
                console.error('❌ Error loading OBJ file:', error);
                reject(error);
              }
            );
          });
        }

        if (!isMounted) return;

        // Step 2.5: Fix Z coordinate deviation by negating all Z coordinates
        // Note: This matches the Z correction applied during coordinate analysis
        console.log('🔧 Applying Z coordinate correction (negating Z values)...');
        object.traverse((child) => {
          if (child.isMesh && child.geometry) {
            const geometry = child.geometry;
            const positionAttribute = geometry.getAttribute('position');
            
            if (positionAttribute) {
              // Negate all Z coordinates to match the coordinate analysis correction
              for (let i = 0; i < positionAttribute.count; i++) {
                const z = positionAttribute.getZ(i);
                positionAttribute.setZ(i, -z);
              }
              positionAttribute.needsUpdate = true;
              geometry.computeBoundingBox();
              geometry.computeBoundingSphere();
              console.log(`   └─ Corrected Z coordinates for ${positionAttribute.count} vertices (matching coordinate analysis)`);
            }
          }
        });

        // Step 3: Position model using EPSG:30169 coordinates
        // IMPORTANT: No rotation or scaling transformations, only position
        const box = new THREE.Box3().setFromObject(object);
        const originalCenter = box.getCenter(new THREE.Vector3());
        const originalSize = box.getSize(new THREE.Vector3());
        
        // Apply scale and mirroring
        object.scale.set(
          scale * (mirrorX ? -1 : 1),  // Apply X mirroring
          scale * (mirrorY ? -1 : 1),  // Apply Y mirroring  
          scale * (mirrorZ ? -1 : 1)   // Apply Z mirroring
        );
        
        // Optional: Mirror the object (uncomment the axis you want to mirror)
        // object.scale.x *= -1;  // Mirror along X-axis (left-right flip)
        // object.scale.y *= -1;  // Mirror along Y-axis (up-down flip)
        // object.scale.z *= -1;  // Mirror along Z-axis (front-back flip)
        
        // Recalculate after scaling
        const scaledBox = new THREE.Box3().setFromObject(object);
        const scaledCenter = scaledBox.getCenter(new THREE.Vector3());
        
        // Position the model: 
        // - Center it horizontally (X and Z axes)
        // - Place bottom at ground level (Y axis)
        // - Maintain original coordinate system orientation
        object.position.set(
          -scaledCenter.x,     // Center X
          -scaledBox.min.y,    // Ground Y (bottom at Y=0)
          -scaledCenter.z      // Center Z
        );
        
        console.log('🏗️ Model positioning (coordinate-based):', {
          epsg30169Center: [coordinateAnalysis.center30169.x, coordinateAnalysis.center30169.z],
          wgs84Position: modelOrigin,
          originalCenter: { x: originalCenter.x.toFixed(2), y: originalCenter.y.toFixed(2), z: originalCenter.z.toFixed(2) },
          scaledCenter: { x: scaledCenter.x.toFixed(2), y: scaledCenter.y.toFixed(2), z: scaledCenter.z.toFixed(2) },
          originalSize: { x: originalSize.x.toFixed(2), y: originalSize.y.toFixed(2), z: originalSize.z.toFixed(2) },
          finalPosition: { x: object.position.x.toFixed(2), y: object.position.y.toFixed(2), z: object.position.z.toFixed(2) },
          scale: scale
        });

        // Step 4: Apply materials with proper lighting support
        console.log('🎨 Configuring materials for proper lighting...');
        let materialCount = 0;
        
        object.traverse((child) => {
          if (child.isMesh) {
            const geometry = child.geometry;
            geometry.computeVertexNormals();

            // If MTL materials were loaded, keep them; otherwise apply default material
            if (mtlPath && child.material) {
              // MTL materials were loaded - keep them and just configure for proper lighting
              const hasVertexColors = geometry.hasAttribute('color');
              console.log(`   ├─ Using MTL material: ${child.material.name || 'unnamed'} (${child.material.constructor.name})${hasVertexColors ? ' with vertex colors' : ''}`);

              // Ensure MTL materials support double-sided rendering
              child.material.side = THREE.DoubleSide;

              // Enable vertex colors if the geometry has them
              if (hasVertexColors) {
                child.material.vertexColors = true;
              }

              // MTL materials are typically MeshPhongMaterial - they work well with lighting
              // No need to convert unless there's a specific issue
              materialCount++;

            } else {
              // No MTL materials - check if geometry has vertex colors
              const hasVertexColors = geometry.hasAttribute('color');

              if (hasVertexColors) {
                console.log(`   ├─ Mesh ${materialCount + 1} has vertex colors - preserving them`);
              } else {
                console.log(`   ├─ Applying default PBR material to mesh ${materialCount + 1}`);
              }

              const material = new THREE.MeshStandardMaterial({
                color: 0xffffff,          // White base color
                roughness: 0.4,           // Slightly rough surface
                metalness: 0.05,          // Almost non-metallic
                side: THREE.DoubleSide,   // Render both sides
                flatShading: false,       // Use smooth shading
                transparent: false,       // Fully opaque
                opacity: 1.0,             // Full opacity
                vertexColors: hasVertexColors  // Enable vertex colors if available
              });
              child.material = material;
              materialCount++;
            }
            
            // Enable shadows for all meshes
            child.castShadow = true;
            child.receiveShadow = true;
          }
        });
        
        console.log(`   └─ Configured ${materialCount} materials for lighting`);

        scene.add(object);

        // Step 5: Add comprehensive lighting setup - ALWAYS ensure proper lighting
        console.log('💡 Setting up comprehensive lighting...');
        
        // Clear any existing lights to avoid conflicts
        const existingLights = scene.children.filter(child => child.isLight);
        existingLights.forEach(light => {
          scene.remove(light);
          console.log(`   ├─ Removed existing light: ${light.type}`);
        });

        // Add ambient light for overall illumination
        const ambientLight = new THREE.AmbientLight(0xffffff, 1.8); // Increased intensity for better overall lighting
        ambientLight.name = 'obj-ambient-light';
        scene.add(ambientLight);
        console.log('   ├─ Added ambient light (intensity: 1.2)');

        // Add primary directional light (sun-like) - 45 degrees above from upper-right-back for better house illumination
        const directionalLight = new THREE.DirectionalLight(0xffffff, 1.8); // Much stronger intensity
        // Position at upper-right-back: positive X (right), positive Y (up), positive Z (back)
        directionalLight.position.set(100, 141, 100); // Y = sqrt(X² + Z²) for 45° angle
        directionalLight.target.position.set(0, 0, 0);
        directionalLight.castShadow = true;
        directionalLight.shadow.mapSize.width = 512; // Higher shadow resolution
        directionalLight.shadow.mapSize.height = 512;
        directionalLight.shadow.camera.near = 0.5;
        directionalLight.shadow.camera.far = 500; // Increased range
        directionalLight.shadow.camera.left = -2000; // Wider shadow coverage
        directionalLight.shadow.camera.right = 2000;
        directionalLight.shadow.camera.top = 2000;
        directionalLight.shadow.camera.bottom = -2000;
        directionalLight.shadow.bias = -0.0001; // Reduce shadow acne
        directionalLight.name = 'obj-directional-light-main';
        scene.add(directionalLight);
        scene.add(directionalLight.target);
        console.log('   ├─ Added main directional light from upper-right-back at 45° (intensity: 1.8)');

        // Add secondary directional light from opposite angle (lower-left-front) for balanced illumination
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 1.0); // Strong secondary light
        directionalLight2.position.set(-100, 141, -100); // Opposite angle: lower-left-front
        directionalLight2.target.position.set(0, 0, 0);
        directionalLight2.castShadow = false; // Disable shadows for secondary light to avoid conflicts
        directionalLight2.name = 'obj-directional-light-secondary';
        scene.add(directionalLight2);
        console.log('   ├─ Added secondary directional light from lower-left-front (intensity: 1.0)');

        // Add overhead light specifically for roof illumination
        const overheadLight = new THREE.DirectionalLight(0xffffff, 0.8);
        overheadLight.position.set(0, 2000, 0); // Directly above
        overheadLight.target.position.set(0, 0, 0);
        overheadLight.castShadow = false; // No shadows to avoid conflicts
        overheadLight.name = 'obj-overhead-light';
        scene.add(overheadLight);
        console.log('   ├─ Added overhead light for roof illumination (intensity: 0.8)');

        // Add hemisphere light for natural outdoor lighting
        const hemisphereLight = new THREE.HemisphereLight(0x87CEEB, 0x8B4513, 0.6); // Increased intensity
        hemisphereLight.name = 'obj-hemisphere-light';
        scene.add(hemisphereLight);
        console.log('   ├─ Added hemisphere light (sky/ground lighting, intensity: 0.6)');

        // Verify lighting setup
        const lights = scene.children.filter(child => child.isLight);
        console.log(`   └─ Total lights in scene: ${lights.length}`);
        lights.forEach(light => {
          console.log(`      ├─ ${light.type}: ${light.name} (intensity: ${light.intensity || 'N/A'})`);
        });

        // Step 6: Add debugging visualization if enabled
        if (showBoundingBox) {
          const finalBox = new THREE.Box3().setFromObject(object);
          const boxHelper = new THREE.Box3Helper(finalBox, 0xff0000);
          scene.add(boxHelper);
          console.log('📦 Added OBJ bounding box visualization (red)');
        }

        // Step 7: Setup custom layer
        const customLayer = {
          id: `obj-model-${Date.now()}`,
          type: 'custom',
          renderingMode: '3d',
          
          onAdd: function (map, gl) {
            console.log('🎭 Custom layer added to map');
          },
          
          render: function (gl, matrix) {
            if (!isMounted) return;
            
            // Apply transformations around the object center, not world origin
            const rotationX = new THREE.Matrix4().makeRotationAxis(
              new THREE.Vector3(1, 0, 0),
              Math.PI / 2
            );
            const rotationZ = new THREE.Matrix4().makeRotationAxis(
              new THREE.Vector3(0, 0, 1),
              Math.PI
            );

            const m = new THREE.Matrix4().fromArray(matrix);
            
            // Create transformation matrix that applies rotations around object center
            // Strategy: translate to origin → rotate → translate back → apply world transforms
            const l = new THREE.Matrix4()
              // 4. Final translation to geographic position
              .makeTranslation(modelTransform.translateX, modelTransform.translateY, modelTransform.translateZ)
              // 3. Apply Mercator scaling (with optional mirroring)
              .scale(new THREE.Vector3(
                -modelTransform.scale,      // X: positive = normal, negative = mirror
                modelTransform.scale,     // Y: negative for Mapbox coordinate system
                -modelTransform.scale       // Z: positive = normal, negative = mirror
              ))
              // Optional: Add additional mirroring here if needed
              // .scale(new THREE.Vector3(-1, 1, 1))  // Mirror X
              // .scale(new THREE.Vector3(1, 1, -1))  // Mirror Z
              // 2. Apply coordinate system rotations (these will now be around object center)
              .multiply(rotationX)
              .multiply(rotationZ);
              // 1. Object is already centered at origin from Stage 1 positioning

            camera.projectionMatrix = m.multiply(l);
            renderer.resetState();
            renderer.render(scene, camera);
          }
        };

        map.addLayer(customLayer);
        customLayerRef.current = customLayer;

        // Step 8: Verify complete setup
        console.log('🔍 Verifying OBJ model setup...');
        console.log(`   ├─ Scene children: ${scene.children.length}`);
        console.log(`   ├─ Lights in scene: ${scene.children.filter(child => child.isLight).length}`);
        console.log(`   ├─ Objects in scene: ${scene.children.filter(child => child.isMesh || child.isGroup).length}`);
        console.log(`   ├─ Renderer shadows enabled: ${renderer.shadowMap.enabled}`);
        console.log(`   ├─ Renderer physical lights: ${renderer.physicallyCorrectLights}`);
        console.log(`   └─ Custom layer ID: ${customLayer.id}`);

        // Add global debugging function for lighting verification
        if (typeof window !== 'undefined') {
          window.debugObjLighting = () => {
            console.log('=== OBJ LIGHTING DEBUG ===');
            console.log('Scene:', scene);
            console.log('Total scene children:', scene.children.length);
            
            const lights = scene.children.filter(child => child.isLight);
            console.log('Lights:', lights.length);
            lights.forEach((light, index) => {
              console.log(`  Light ${index + 1}:`, {
                type: light.type,
                name: light.name,
                intensity: light.intensity,
                color: `#${light.color.getHexString()}`,
                position: light.position,
                visible: light.visible
              });
            });
            
            const meshes = [];
            scene.traverse((child) => {
              if (child.isMesh) {
                meshes.push({
                  name: child.name || 'unnamed',
                  material: child.material.type,
                  castShadow: child.castShadow,
                  receiveShadow: child.receiveShadow,
                  visible: child.visible
                });
              }
            });
            console.log('Meshes:', meshes.length);
            meshes.forEach((mesh, index) => {
              console.log(`  Mesh ${index + 1}:`, mesh);
            });
            
            console.log('Renderer settings:', {
              shadowMapEnabled: renderer.shadowMap.enabled,
              shadowMapType: renderer.shadowMap.type,
              physicallyCorrectLights: renderer.physicallyCorrectLights,
              toneMapping: renderer.toneMapping,
              outputEncoding: renderer.outputEncoding
            });
            console.log('========================');
          };
          
          console.log('💡 Debug function available: window.debugObjLighting()');
        }

        // Callback with analysis data
        if (onModelLoaded) {
          onModelLoaded({
            coordinateAnalysis,
            object,
            scene,
            modelOrigin,
            dimensions: coordinateAnalysis.dimensions
          });
        }

        console.log('✅ OBJ model loaded and positioned using coordinates');

      } catch (error) {
        console.error('❌ Failed to load OBJ model with coordinates:', error);
      }
    };

    loadModelWithCoordinates();

    // Cleanup
    return () => {
      isMounted = false;
      if (customLayerRef.current && map.getLayer(customLayerRef.current.id)) {
        console.log('🧹 Cleaning up OBJ model layer');
        map.removeLayer(customLayerRef.current.id);
        customLayerRef.current = null;
      }
    };
  }, [map, objPath, mtlPath, scale, mirrorX, mirrorY, mirrorZ, showBoundingBox, onModelLoaded]);
  // This component doesn't render DOM elements
  return null;
};

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Compare primitive props
  if (prevProps.objPath !== nextProps.objPath) return false;
  if (prevProps.mtlPath !== nextProps.mtlPath) return false;
  if (prevProps.altitude !== nextProps.altitude) return false;
  if (prevProps.scale !== nextProps.scale) return false;
  if (prevProps.mirrorX !== nextProps.mirrorX) return false;
  if (prevProps.mirrorY !== nextProps.mirrorY) return false;
  if (prevProps.mirrorZ !== nextProps.mirrorZ) return false;
  if (prevProps.showBoundingBox !== nextProps.showBoundingBox) return false;
  if (prevProps.showGeojson !== nextProps.showGeojson) return false;
  if (prevProps.showObjModel !== nextProps.showObjModel) return false;

  // Compare object/array props by reference
  if (prevProps.map !== nextProps.map) return false;
  if (prevProps.visibleLod3Layers !== nextProps.visibleLod3Layers) return false;

  // Compare callbacks
  if (prevProps.onModelLoaded !== nextProps.onModelLoaded) return false;
  if (prevProps.hideBuildingsInGeojsonArea !== nextProps.hideBuildingsInGeojsonArea) return false;
  if (prevProps.clearBuildingFilters !== nextProps.clearBuildingFilters) return false;

  return true;
};

export default React.memo(ObjModelViewer, arePropsEqual);