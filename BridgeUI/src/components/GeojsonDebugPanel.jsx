import React, { useState, useRef, useEffect } from 'react';
import { getHeightStatistics, validateGeojsonData } from '../utils/geojsonUtils';

const GeojsonDebugPanel = ({ geojsonData, isVisible }) => {
  const [position, setPosition] = useState({ x: 16, y: 76 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const panelRef = useRef(null);

  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging) return;
      
      const newX = e.clientX - dragOffset.x;
      const newY = e.clientY - dragOffset.y;
      
      // Keep panel within viewport bounds
      const maxX = window.innerWidth - 300; // panel width
      const maxY = window.innerHeight - 400; // approximate panel height
      
      setPosition({
        x: Math.max(0, Math.min(newX, maxX)),
        y: Math.max(0, Math.min(newY, maxY))
      });
    };

    const handleMouseUp = () => {
      setIsDragging(false);
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, dragOffset]);

  const handleMouseDown = (e) => {
    if (!panelRef.current) return;
    
    const rect = panelRef.current.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
    setIsDragging(true);
  };

  if (!isVisible || !geojsonData) {
    return null;
  }

  const stats = getHeightStatistics(geojsonData);
  const validation = validateGeojsonData(geojsonData);

  return (
    <div 
      ref={panelRef}
      className={`geojson-debug-panel ${isDragging ? 'dragging' : ''}`}
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`
      }}
      onMouseDown={handleMouseDown}
    >
      <h4>GeoJSON Debug Info</h4>
      
      <div className="debug-section">
        <h5>Height Statistics</h5>
        {stats && (
          <div className="stats-grid">
            <div>Features: {stats.count}</div>
            <div>Min Height: {stats.min.toFixed(1)}m</div>
            <div>Max Height: {stats.max.toFixed(1)}m</div>
            <div>Average: {stats.average.toFixed(1)}m</div>
            <div>Median: {stats.median.toFixed(1)}m</div>
            <div>Range: {stats.range.toFixed(1)}m</div>
          </div>
        )}
      </div>

      <div className="debug-section">
        <h5>Validation</h5>
        <div>Total Features: {validation.stats.totalFeatures}</div>
        <div>With predHeight: {validation.stats.featuresWithPredHeight}</div>
        <div>Without height: {validation.stats.featuresWithoutHeight}</div>
        
        {validation.warnings.length > 0 && (
          <div className="warnings">
            <strong>Warnings:</strong>
            <ul>
              {validation.warnings.map((warning, index) => (
                <li key={index}>{warning}</li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="debug-section">
        <h5>Sample Features</h5>
        {geojsonData.features.slice(0, 3).map((feature, index) => (
          <div key={index} className="feature-sample">
            <strong>Feature {index}:</strong>
            <div>ID: {feature.properties?.id || 'No ID'}</div>
            <div>predHeight: {feature.properties?.predHeight || 'Not set'}</div>
            <div>height: {feature.properties?.height || 'Not set'}</div>
            <div>area: {feature.properties?.area?.toFixed(2) || 'Not set'}</div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default GeojsonDebugPanel;
