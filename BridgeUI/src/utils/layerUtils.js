/**
 * Unified Layer Management System
 * Handles all layer types (geojson, lod1, lod2, lod3, orthophoto) with proper relationships
 */

/**
 * Generate a unique ID based on filename and timestamp
 * @param {string} filename - Original filename
 * @param {string} prefix - Prefix for the ID (e.g., 'lod1', 'geojson')
 * @returns {string} - Unique ID
 */
export const generateUniqueId = (filename, prefix = '') => {
  // Clean the filename - remove extension and special characters
  const cleanName = filename
    .replace(/\.[^/.]+$/, '') // Remove file extension
    .replace(/[^a-zA-Z0-9_-]/g, '_') // Replace special chars with underscore
    .toLowerCase();
  
  // Generate timestamp-based suffix
  const timestamp = Date.now();
  const randomSuffix = Math.random().toString(36).substring(2, 6);
  
  // Combine parts
  const parts = [prefix, cleanName, timestamp, randomSuffix].filter(Boolean);
  return parts.join('-');
};

/**
 * Create a unified layer structure for all layer types
 * @param {string} id - Unique layer ID
 * @param {string} type - Layer type ('geojson', 'lod1', 'lod2', 'lod3', 'orthophoto')
 * @param {string} name - Display name
 * @param {Object} data - Layer data
 * @param {string} geojsonId - ID of the corresponding GeoJSON layer (for LoD models)
 * @param {Object} metadata - Additional metadata
 * @returns {Object} - Unified layer object
 */
export const createUnifiedLayer = (id, type, name, data, geojsonId = null, metadata = {}) => {
  return {
    id,
    type, // 'geojson', 'lod1', 'lod2', 'lod3', 'orthophoto'
    name,
    data,
    geojsonId, // Links LoD models to their source GeoJSON
    metadata: {
      ...metadata,
      createdAt: new Date().toISOString(),
      lastModified: new Date().toISOString()
    },
    visible: true,
    controlId: generateControlId(id, type) // Unique control identifier
  };
};

/**
 * Create control configuration for layer controls
 * @param {string} layerId - Layer ID
 * @param {string} type - Layer type
 * @param {string} instanceId - Specific instance ID for the control
 * @returns {Object} - Control configuration
 */
export const createLayerControl = (layerId, type, instanceId) => {
  return {
    id: `control-${layerId}-${instanceId}`,
    type,
    instanceId,
    layerId,
    timestamp: Date.now()
  };
};

/**
 * Extract display name from filename
 * @param {string} filename - Original filename
 * @returns {string} - Clean display name
 */
export const getDisplayName = (filename) => {
  return filename
    .replace(/\.[^/.]+$/, '') // Remove file extension
    .replace(/[_-]/g, ' ') // Replace underscores and hyphens with spaces
    .replace(/\b\w/g, l => l.toUpperCase()); // Capitalize first letter of each word
};

/**
 * Layer Manager Class - Central management for all layers
 */
export class LayerManager {
  constructor() {
    this.layers = new Map(); // Store all layers by ID
    this.controls = new Map(); // Store control configurations
    this.typeGroups = new Map(); // Group layers by type
    this.geojsonRelations = new Map(); // Map LoD models to their GeoJSON sources
  }

  /**
   * Add a new layer to the manager
   * @param {Object} layerData - Layer data
   * @returns {string} - Layer ID
   */
  addLayer(layerData) {
    const layer = {
      ...layerData,
      controlId: generateControlId(layerData.id, layerData.type),
      // Ensure metadata always exists
      metadata: layerData.metadata || {}
    };
    
    this.layers.set(layer.id, layer);
    
    // Group by type
    if (!this.typeGroups.has(layer.type)) {
      this.typeGroups.set(layer.type, new Set());
    }
    this.typeGroups.get(layer.type).add(layer.id);
    
    // Track GeoJSON relationships for LoD models
    if (layer.geojsonId && layer.type.startsWith('lod')) {
      if (!this.geojsonRelations.has(layer.geojsonId)) {
        this.geojsonRelations.set(layer.geojsonId, new Set());
      }
      this.geojsonRelations.get(layer.geojsonId).add(layer.id);
    }
    
    console.log(`📝 Added layer: ${layer.id} (type: ${layer.type}, geojsonId: ${layer.geojsonId})`);
    return layer.id;
  }

  /**
   * Remove a layer from the manager
   * @param {string} layerId - Layer ID to remove
   */
  removeLayer(layerId) {
    const layer = this.layers.get(layerId);
    if (!layer) return false;
    
    // Remove from type groups
    if (this.typeGroups.has(layer.type)) {
      this.typeGroups.get(layer.type).delete(layerId);
    }
    
    // Remove from GeoJSON relations
    if (layer.geojsonId && this.geojsonRelations.has(layer.geojsonId)) {
      this.geojsonRelations.get(layer.geojsonId).delete(layerId);
    }
    
    // Remove the layer
    this.layers.delete(layerId);
    this.controls.delete(layerId);
    
    console.log(`🗑️ Removed layer: ${layerId}`);
    return true;
  }

  /**
   * Get layers by type
   * @param {string} type - Layer type
   * @returns {Array} - Array of layers
   */
  getLayersByType(type) {
    const layerIds = this.typeGroups.get(type) || new Set();
    return Array.from(layerIds).map(id => this.layers.get(id)).filter(Boolean);
  }

  /**
   * Get layers related to a specific GeoJSON
   * @param {string} geojsonId - GeoJSON layer ID
   * @returns {Array} - Array of related layers
   */
  getRelatedLayers(geojsonId) {
    const relatedIds = this.geojsonRelations.get(geojsonId) || new Set();
    return Array.from(relatedIds).map(id => this.layers.get(id)).filter(Boolean);
  }

  /**
   * Get all layers as array
   * @returns {Array} - All layers
   */
  getAllLayers() {
    return Array.from(this.layers.values());
  }

  /**
   * Get layer by ID
   * @param {string} layerId - Layer ID
   * @returns {Object|null} - Layer or null
   */
  getLayer(layerId) {
    return this.layers.get(layerId) || null;
  }

  /**
   * Update layer visibility
   * @param {string} layerId - Layer ID
   * @param {boolean} visible - Visibility state
   */
  setLayerVisibility(layerId, visible) {
    const layer = this.layers.get(layerId);
    if (layer) {
      layer.visible = visible;
      // Ensure metadata exists before setting properties
      if (!layer.metadata) {
        layer.metadata = {};
      }
      layer.metadata.lastModified = new Date().toISOString();
    }
  }

  /**
   * Get organized layers for LayerPanel
   * @returns {Object} - Organized layers by type
   */
  getOrganizedLayers() {
    return {
      geojson: this.getLayersByType('geojson'),
      lod1: this.getLayersByType('lod1'),
      lod2: this.getLayersByType('lod2'), 
      lod3: this.getLayersByType('lod3'),
      orthophoto: this.getLayersByType('orthophoto'),
      custom: this.getLayersByType('custom') // For backward compatibility
    };
  }

  /**
   * Clear all layers
   */
  clearAllLayers() {
    this.layers.clear();
    this.controls.clear();
    this.typeGroups.clear();
    this.geojsonRelations.clear();
    console.log('🧹 Cleared all layers');
  }
}

/**
 * Generate unique control ID for layer controls
 * @param {string} layerId - Layer ID
 * @param {string} controlType - Type of control ('visibility', 'zoom', 'style')
 * @returns {string} - Unique control ID
 */
export const generateControlId = (layerId, controlType = 'main') => {
  return `${layerId}-control-${controlType}-${Date.now()}`;
};

/**
 * Extract metadata from filename and file object
 * @param {File} file - File object
 * @returns {Object} - Extracted metadata
 */
export const extractFileMetadata = (file) => {
  return {
    originalName: file.name,
    size: file.size,
    type: file.type,
    lastModified: new Date(file.lastModified).toISOString(),
    uploadedAt: new Date().toISOString()
  };
};

/**
 * Validate layer ID uniqueness
 * @param {string} layerId - Layer ID to validate
 * @param {Array} existingLayers - Array of existing layers
 * @returns {boolean} - True if unique, false if duplicate
 */
export const validateLayerIdUniqueness = (layerId, existingLayers) => {
  return !existingLayers.some(layer => layer.id === layerId);
};

/**
 * Get layer type from file extension or data
 * @param {string} filename - Filename
 * @param {Object} data - File data (optional)
 * @returns {string} - Detected layer type
 */
export const detectLayerType = (filename, data = null) => {
  const extension = filename.toLowerCase().split('.').pop();
  
  switch (extension) {
    case 'geojson':
    case 'json':
      return 'geojson';
    case 'obj':
      return 'model';
    case 'jpg':
    case 'jpeg':
    case 'png':
    case 'tiff':
    case 'tif':
      return 'orthophoto';
    case 'las':
    case 'laz':
    case 'ply':
      return 'pointcloud';
    default:
      return 'unknown';
  }
};

/**
 * Generate LoD1 model name from GeoJSON data
 * @param {string} geojsonName - Original GeoJSON name
 * @param {Object} lodConfig - LoD configuration
 * @returns {string} - LoD1 model name
 */
export const generateLoDModelName = (geojsonName, lodConfig = {}) => {
  const baseName = getDisplayName(geojsonName);
  const height = lodConfig.height ? ` (${lodConfig.height}m)` : '';
  const timestamp = new Date().toLocaleTimeString('en-US', { 
    hour12: false, 
    hour: '2-digit', 
    minute: '2-digit' 
  });
  
  return `${baseName} LoD1${height} - ${timestamp}`;
};
