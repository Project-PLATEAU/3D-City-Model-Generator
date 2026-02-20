import React, { useState, useEffect } from 'react';
import '../styles/Components.css';

const ConfigurationPanel = ({ 
  selectedPolygon, 
  onConfigurationUpdate, 
  onGenerateLoD1,
  onClose 
}) => {
  const [config, setConfig] = useState({
    height: '',
    roofType: 'flat'
  });

  // Update config when a new polygon is selected
  useEffect(() => {
    if (selectedPolygon && selectedPolygon.properties) {
      setConfig({
        height: selectedPolygon.properties.predHeight || selectedPolygon.properties.height || '',
        roofType: 'flat' // LoD1 only supports flat roofs
      });
    }
  }, [selectedPolygon]);

  // Handle ESC key to exit configuration mode
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && onClose) {
        onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  const handleHeightChange = (e) => {
    const newHeight = e.target.value;
    setConfig(prev => ({ ...prev, height: newHeight }));
    
    // Update the polygon configuration immediately
    if (selectedPolygon && onConfigurationUpdate) {
      const updatedConfig = { ...config, height: newHeight };
      onConfigurationUpdate(selectedPolygon.properties?.id || selectedPolygon.id, updatedConfig);
    }
  };

  const handleGenerateLoD1 = () => {
    if (onGenerateLoD1) {
      onGenerateLoD1();
    }
  };

  if (!selectedPolygon) {
    return (
      <div className="configuration-panel">
        <div className="configuration-header">
          <h3>パラメータ</h3>
          <button 
            className="close-button-round" 
            onClick={onClose}
            title="Close Configuration (ESC)"
          >
            ×
          </button>
        </div>
        <div className="configuration-content">
          <p style={{ textAlign: 'center', color: '#666', margin: '20px 0' }}>
            建物をクリックしてパラメータを編集
          </p>
        </div>
        <button
            className="generate-button"
            onClick={handleGenerateLoD1}
            style={{
              backgroundColor: '#8BC34A',
            }}
          >
            LoD1モデルを再生成
          </button>
      </div>
    );
  }

  const polygonId = selectedPolygon.properties?.id || selectedPolygon.id || 'Unknown';

  return (
    <div className="configuration-panel">
      <div className="configuration-header">
        <h3>CONFIGURATION</h3>
        <button 
          className="close-button-round" 
          onClick={onClose}
          title="Close Configuration (ESC)"
        >
          ×
        </button>
      </div>
      
      <div className="configuration-content">
        <div className="config-field">
          <label htmlFor="polygon-id">建物ID：</label>
          <input
            id="polygon-id"
            type="text"
            value={polygonId}
            readOnly
            className="config-input readonly"
          />
        </div>

        <div className="config-field">
          <label htmlFor="roof-type">屋根タイプ：</label>
          <input
            id="roof-type"
            type="text"
            value="Flat"
            readOnly
            className="config-input readonly"
            title="Roof type is frozen for LoD1"
          />
          <small className="field-note">LoD1では選択不可</small>
        </div>

        <div className="config-field">
          <label htmlFor="height">高さ（m）：</label>
          <input
            id="height"
            type="number"
            value={config.height}
            onChange={handleHeightChange}
            placeholder="Enter height"
            className="config-input"
            min="0"
            step="0.1"
          />
        </div>

        <div className="config-actions">
          <button
            className="generate-button"
            onClick={handleGenerateLoD1}
            disabled={!config.height}
            style={{
              backgroundColor: config.height ? '#8BC34A' : '#ccc',
              cursor: config.height ? 'pointer' : 'not-allowed'
            }}
          >
            LoD1モデルを再生成
          </button>
        </div>
      </div>
    </div>
  );
};

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Compare object props by reference
  if (prevProps.selectedPolygon !== nextProps.selectedPolygon) return false;

  // Compare callbacks (should be memoized with useCallback in parent)
  if (prevProps.onConfigurationUpdate !== nextProps.onConfigurationUpdate) return false;
  if (prevProps.onGenerateLoD1 !== nextProps.onGenerateLoD1) return false;
  if (prevProps.onClose !== nextProps.onClose) return false;

  return true;
};

export default React.memo(ConfigurationPanel, arePropsEqual);
