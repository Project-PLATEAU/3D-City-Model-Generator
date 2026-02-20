import proj4 from 'proj4';

/**
 * Coordinate transformation utilities for converting between EPSG:30169 and EPSG:4326
 * 
 * Note: Mapbox GL JS expects coordinates in [longitude, latitude] order (OGC:CRS84 style),
 * not [latitude, longitude] as specified in newer EPSG:4326 definitions.
 */

// Ensure EPSG:30169 is defined (Tokyo / Japan Plane Rectangular CS IX)
if (!proj4.defs['EPSG:30169']) {
  // EPSG:30169 - Tokyo / Japan Plane Rectangular CS IX
  // Original definition with Tokyo datum, transforming to WGS84
  proj4.defs('EPSG:30169', '+proj=tmerc +lat_0=36 +lon_0=139.8333333333333 +k=0.9999 +x_0=0 +y_0=0 +ellps=bessel +towgs84=-146.414,507.337,680.507,0,0,0,0 +units=m +no_defs');
  console.log('🗾 Defined EPSG:30169 projection with Tokyo datum and WGS84 transformation parameters');
}

/**
 * Convert coordinates from EPSG:30169 to EPSG:4326 (WGS84)
 * @param {number} x - X coordinate in EPSG:30169 (meters)
 * @param {number} z - Z coordinate in EPSG:30169 (meters) 
 * @returns {[number, number]} - [longitude, latitude] in EPSG:4326 for Mapbox
 */
export function convertEPSG30169ToWGS84(x, z) {
  try {
    // Convert from EPSG:30169 to EPSG:4326
    const [lon, lat] = proj4('EPSG:30169', 'EPSG:4326', [x, z]);
    
    // Validate the result
    if (typeof lon !== 'number' || typeof lat !== 'number' || 
        isNaN(lon) || isNaN(lat) ||
        lon < -180 || lon > 180 || lat < -90 || lat > 90) {
      throw new Error(`Invalid converted coordinates: [${lon}, ${lat}]`);
    }
    
    console.log(`🔄 Converted EPSG:30169 [${x.toFixed(2)}, ${z.toFixed(2)}] → WGS84 [${lon.toFixed(6)}, ${lat.toFixed(6)}]`);
    
    // Return in Mapbox format: [longitude, latitude]
    return [lon, lat];
  } catch (error) {
    console.error('❌ Coordinate conversion failed:', error);
    console.error(`   Input: EPSG:30169 [${x}, ${z}]`);
    
    // Return a fallback position in Tokyo area
    return [139.7, 35.7];
  }
}

/**
 * Extract the bounding box from OBJ file vertices and convert to WGS84
 * @param {string} objContent - Raw OBJ file content
 * @returns {Object} - Bounding box and center in both coordinate systems
 */
export function extractObjBoundsAndConvert(objContent) {
  console.log(`🔍 DEBUG: Starting OBJ analysis`);
  console.log(`   Content length: ${objContent?.length || 'undefined'}`);
  console.log(`   Content type: ${typeof objContent}`);
  
  if (!objContent || typeof objContent !== 'string') {
    console.error('❌ Invalid objContent:', objContent);
    return null;
  }
  
  console.log(`   First 200 chars: "${objContent.substring(0, 200)}"`);
  
  const lines = objContent.split('\n');
  console.log(`   Total lines: ${lines.length}`);
  console.log(`   First 5 lines:`, lines.slice(0, 5));
  
  const vertices = [];
  let vertexLineCount = 0;
  
  // Extract all vertices from OBJ file
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith('v ')) {
      vertexLineCount++;
      
      if (vertexLineCount <= 3) {
        console.log(`   Processing vertex line ${vertexLineCount}: "${line}"`);
      }
      
      const parts = line.split(/\s+/);
      const x = parseFloat(parts[1]);
      const y = parseFloat(parts[2]); // Height (Y-up in OBJ)
      let z = parseFloat(parts[3]);
      
      if (vertexLineCount <= 3) {
        console.log(`   Parsed: x=${x}, y=${y}, z=${z}`);
        console.log(`   Valid: x=${!isNaN(x)}, y=${!isNaN(y)}, z=${!isNaN(z)}`);
      }
      
      // Apply Z coordinate correction here during loading, @LIAO please remove when the issue with OBJ export is resolved
      z = -z; // Negate Z to fix location deviation
      
      if (!isNaN(x) && !isNaN(y) && !isNaN(z)) {
        vertices.push({ x, y, z });
      }
    }
  }
  
  console.log(`📊 Found ${vertexLineCount} vertex lines, parsed ${vertices.length} vertices`);
  
  if (vertices.length === 0) {
    console.warn('⚠️ No vertices found in OBJ file');
    console.warn('🔍 First 10 lines for debugging:');
    lines.slice(0, 10).forEach((line, i) => {
      console.warn(`   Line ${i}: "${line}" (length: ${line.length})`);
    });
    return null;
  }
  
  // Calculate bounding box in EPSG:30169 (with corrected Z coordinates)
  const bounds30169 = {
    minX: Math.min(...vertices.map(v => v.x)),
    maxX: Math.max(...vertices.map(v => v.x)),
    minY: Math.min(...vertices.map(v => v.y)),
    maxY: Math.max(...vertices.map(v => v.y)),
    minZ: Math.min(...vertices.map(v => v.z)),
    maxZ: Math.max(...vertices.map(v => v.z))
  };
  
  // Calculate center in EPSG:30169 (with corrected Z coordinates)
  const center30169 = {
    x: (bounds30169.minX + bounds30169.maxX) / 2,
    y: (bounds30169.minY + bounds30169.maxY) / 2,
    z: (bounds30169.minZ + bounds30169.maxZ) / 2
  };
  
  // Convert center to WGS84 for Mapbox positioning (using corrected Z)
  const centerWGS84 = convertEPSG30169ToWGS84(center30169.x, center30169.z);
  
  // Convert corner points for validation (using corrected Z)
  const corners = [
    convertEPSG30169ToWGS84(bounds30169.minX, bounds30169.minZ), // bottom-left
    convertEPSG30169ToWGS84(bounds30169.maxX, bounds30169.minZ), // bottom-right
    convertEPSG30169ToWGS84(bounds30169.maxX, bounds30169.maxZ), // top-right
    convertEPSG30169ToWGS84(bounds30169.minX, bounds30169.maxZ), // top-left
  ];
  
  console.log('📐 OBJ Coordinate Analysis (with Z correction):');
  console.log(`   EPSG:30169 bounds: X[${bounds30169.minX.toFixed(2)}, ${bounds30169.maxX.toFixed(2)}] Z[${bounds30169.minZ.toFixed(2)}, ${bounds30169.maxZ.toFixed(2)}]`);
  console.log(`   EPSG:30169 center: [${center30169.x.toFixed(2)}, ${center30169.z.toFixed(2)}]`);
  console.log(`   WGS84 center: [${centerWGS84[0].toFixed(6)}, ${centerWGS84[1].toFixed(6)}]`);
  console.log(`   WGS84 corners:`, corners.map(c => `[${c[0].toFixed(6)}, ${c[1].toFixed(6)}]`));
  console.log(`   🔧 Applied Z negation during coordinate analysis`);
  
  return {
    success: true,
    vertexCount: vertices.length,
    vertices: vertices,
    bounds: bounds30169,
    bounds30169: bounds30169,
    center: center30169,
    center30169: center30169,
    mapboxPosition: centerWGS84,
    centerWGS84: centerWGS84,
    cornersWGS84: corners,
    dimensions: {
      width: bounds30169.maxX - bounds30169.minX,
      height: bounds30169.maxY - bounds30169.minY,
      depth: bounds30169.maxZ - bounds30169.minZ
    }
  };
}

/**
 * Load and analyze OBJ file coordinates
 * @param {string} objPath - Path to the OBJ file
 * @returns {Promise<Object>} - Analysis results with converted coordinates
 */
export async function loadAndAnalyzeObjCoordinates(objPath) {
  try {
    console.log(`📥 Loading OBJ file for coordinate analysis: ${objPath}`);
    
    const response = await fetch(objPath);
    if (!response.ok) {
      throw new Error(`Failed to load OBJ file: ${response.status} ${response.statusText}`);
    }
    
    const objContent = await response.text();
    const analysis = extractObjBoundsAndConvert(objContent);
    
    if (!analysis) {
      throw new Error('Failed to analyze OBJ coordinates');
    }
    
    console.log(`✅ Successfully analyzed OBJ coordinates from: ${objPath}`);
    return analysis;
    
  } catch (error) {
    console.error('❌ Failed to load and analyze OBJ coordinates:', error);
    return {
      success: false,
      error: error.message,
      vertexCount: 0,
      vertices: [],
      bounds: null,
      center: null,
      mapboxPosition: null
    };
  }
}