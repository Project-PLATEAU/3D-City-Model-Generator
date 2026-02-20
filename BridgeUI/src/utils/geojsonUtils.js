/**
 * Utility functions for handling GeoJSON data and height management
 */

/**
 * Process GeoJSON data to ensure proper height properties for visualization
 * @param {Object} geojsonData - The original GeoJSON data
 * @returns {Object} - Processed GeoJSON with normalized height properties
 */
export const processGeojsonHeights = (geojsonData) => {
  if (!geojsonData || !geojsonData.features) {
    return geojsonData;
  }

  return {
    ...geojsonData,
    features: geojsonData.features.map(feature => ({
      ...feature,
      properties: {
        ...feature.properties,
        // Priority: predHeight > height > HEIGHT > default 9m
        height: getFeatureHeight(feature.properties),
        base_height: feature.properties?.base_height || 
                    feature.properties?.BASE_HEIGHT || 
                    0
      }
    }))
  };
};

/**
 * Get the height value for a feature from its properties
 * @param {Object} properties - Feature properties
 * @returns {number} - Height value in meters
 */
export const getFeatureHeight = (properties) => {
  if (!properties) return 9;
  
  // Check predHeight first (this is the configured height)
  if (properties.predHeight !== undefined && properties.predHeight !== null) {
    return parseFloat(properties.predHeight) || 9;
  }
  
  // Fall back to height property
  if (properties.height !== undefined && properties.height !== null) {
    return parseFloat(properties.height) || 9;
  }
  
  // Fall back to HEIGHT property (uppercase)
  if (properties.HEIGHT !== undefined && properties.HEIGHT !== null) {
    return parseFloat(properties.HEIGHT) || 9;
  }
  
  // Default to 9 meters
  return 9;
};

/**
 * Update the height of a specific polygon in GeoJSON data
 * @param {Object} geojsonData - The original GeoJSON data
 * @param {string|number} polygonId - The ID of the polygon to update
 * @param {number} newHeight - The new height value
 * @returns {Object} - Updated GeoJSON data
 */
export const updatePolygonHeight = (geojsonData, polygonId, newHeight) => {
  if (!geojsonData || !geojsonData.features) {
    return geojsonData;
  }

  return {
    ...geojsonData,
    features: geojsonData.features.map(feature => {
      const featureId = feature.properties?.id || feature.id;
      if (featureId === polygonId || featureId === String(polygonId)) {
        return {
          ...feature,
          properties: {
            ...feature.properties,
            predHeight: parseFloat(newHeight) || 9,
            height: parseFloat(newHeight) || 9,
            lastModified: new Date().toISOString()
          }
        };
      }
      return feature;
    })
  };
};

/**
 * Get polygon by ID from GeoJSON data
 * @param {Object} geojsonData - The GeoJSON data
 * @param {string|number} polygonId - The ID of the polygon
 * @returns {Object|null} - The feature or null if not found
 */
export const getPolygonById = (geojsonData, polygonId) => {
  if (!geojsonData || !geojsonData.features) {
    return null;
  }

  return geojsonData.features.find(feature => {
    const featureId = feature.properties?.id || feature.id;
    return featureId === polygonId || featureId === String(polygonId);
  });
};

/**
 * Validate GeoJSON data for required properties
 * @param {Object} geojsonData - The GeoJSON data to validate
 * @returns {Object} - Validation result with warnings and errors
 */
export const validateGeojsonData = (geojsonData) => {
  const result = {
    isValid: true,
    warnings: [],
    errors: [],
    stats: {
      totalFeatures: 0,
      featuresWithPredHeight: 0,
      featuresWithoutHeight: 0,
      averageHeight: 0
    }
  };

  if (!geojsonData) {
    result.isValid = false;
    result.errors.push('No GeoJSON data provided');
    return result;
  }

  if (!geojsonData.features || !Array.isArray(geojsonData.features)) {
    result.isValid = false;
    result.errors.push('GeoJSON data must have a features array');
    return result;
  }

  let totalHeight = 0;
  result.stats.totalFeatures = geojsonData.features.length;

  geojsonData.features.forEach((feature, index) => {
    if (!feature.properties) {
      result.warnings.push(`Feature ${index} has no properties`);
      return;
    }

    const height = getFeatureHeight(feature.properties);
    totalHeight += height;

    if (feature.properties.predHeight !== undefined) {
      result.stats.featuresWithPredHeight++;
    } else if (!feature.properties.height && !feature.properties.HEIGHT) {
      result.stats.featuresWithoutHeight++;
    }

    if (!feature.properties.id && feature.id === undefined) {
      result.warnings.push(`Feature ${index} has no ID property`);
    }
  });

  result.stats.averageHeight = totalHeight / result.stats.totalFeatures;

  if (result.stats.featuresWithoutHeight > 0) {
    result.warnings.push(
      `${result.stats.featuresWithoutHeight} features will use default height of 9m`
    );
  }

  return result;
};

/**
 * Create a summary of height statistics for GeoJSON data
 * @param {Object} geojsonData - The GeoJSON data
 * @returns {Object} - Height statistics
 */
export const getHeightStatistics = (geojsonData) => {
  if (!geojsonData || !geojsonData.features) {
    return null;
  }

  const heights = geojsonData.features.map(feature => 
    getFeatureHeight(feature.properties)
  );

  heights.sort((a, b) => a - b);

  return {
    count: heights.length,
    min: heights[0],
    max: heights[heights.length - 1],
    average: heights.reduce((sum, h) => sum + h, 0) / heights.length,
    median: heights.length % 2 === 0 
      ? (heights[heights.length / 2 - 1] + heights[heights.length / 2]) / 2
      : heights[Math.floor(heights.length / 2)],
    range: heights[heights.length - 1] - heights[0]
  };
};
