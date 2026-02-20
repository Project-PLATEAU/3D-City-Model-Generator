import React from 'react';
import zoomInIcon from '../assets/icons/zoom-in.svg?url';

const LayerPanel = ({ layers, onLayerToggle, visibleLayers, zoomToLayer, zoomToOrthophoto, layerManager }) => {
  const handleLayerToggle = (layerId) => {
    console.log(`🎛️ LayerPanel: Toggle request for ${layerId}`);
    onLayerToggle && onLayerToggle(layerId, !visibleLayers.has(layerId));
  };

  const handleZoomToLayer = (layerId, layerType) => {
    console.log(`🎯 LayerPanel: Zoom request for ${layerId} (type: ${layerType})`);

    // Use appropriate zoom function based on layer type
    if (layerType === 'orthophoto' && zoomToOrthophoto) {
      zoomToOrthophoto(layerId);
    } else if (zoomToLayer) {
      zoomToLayer(layerId);
    }
  };

  // LoD1 legend items (buildings, roads, city furnitures)
  const lod1Legend = [
    { label: '建物', color: '#F6CD4E' },
    { label: '道路', color: '#CECCC2' },
    { label: '都市設備', color: '#C87E7B' }
  ];

  // LoD2 legend items
  const lod2Legend = [
    { label: '建物', color: '#88B0C1' },
    { label: '植生', color: 'rgb(137, 179, 95)' },
    { label: '都市設備', color: '#C87E7B' }, 
    { label: '道路', isRoadGroup: true, roads: [
      { label: '高速道路', color: 'rgb(220, 80, 80)' },
      { label: '国道', color: 'rgb(230, 130, 70)' },
      { label: '主要地方道', color: 'rgb(240, 170, 70)' },
      { label: '一般県道', color: 'rgb(250, 210, 80)' },
      { label: '市町村道', color: 'rgb(250, 240, 140)' },
      { label: '生活道路', color: 'rgb(200, 200, 200)' },
      { label: '歩道', color: 'rgb(120, 200, 120)' },
      { label: '自転車道', color: 'rgb(120, 160, 220)' }
    ]}
  ];

  // LoD2 BMQI legend - gradient from low quality (red) to high quality (white)
  const lod2BMQIGradient = {
    lowLabel: 'ノイズあり（0.95）',
    highLabel: 'ノイズなし（1.0）',
    // Gradient colors matching generation.py: #EB6D7B (low) → #FFFFFF (high)
    gradient: 'linear-gradient(to right, #EB6D7B, #F0A080, #FFFFFF)'
  };

  // LoD3 legend items
  const lod3Legend = [
    { label: '建物', isBuildingGroup: true, parts: [
      { label: '建物本体', color: '#88B0C1' },
      { label: '窓', color: '#0000B7' },
      { label: 'ドア/入口', color: '#CFBE61' },
      { label: 'バルコニー', color: '#C0C1C2' }
    ]},
    { label: '植生', color: 'rgb(137, 179, 95)' },
    { label: '道路', isRoadGroup: true, roads: [
      { label: '高速道路', color: 'rgb(220, 80, 80)' },
      { label: '国道', color: 'rgb(230, 130, 70)' },
      { label: '主要地方道', color: 'rgb(240, 170, 70)' },
      { label: '一般県道', color: 'rgb(250, 210, 80)' },
      { label: '市町村道', color: 'rgb(250, 240, 140)' },
      { label: '生活道路', color: 'rgb(200, 200, 200)' },
      { label: '歩道', color: 'rgb(120, 200, 120)' },
      { label: '自転車道', color: 'rgb(120, 160, 220)' }
    ]},
    { label: '都市設備', color: '#C87E7B' }
  ];

  // Check if there are any visible LoD1 layers
  const lod1Layers = layers.lod1 || [];
  const hasVisibleLod1 = lod1Layers.some(layer => visibleLayers.has(layer.id));

  // Check if there are any visible LoD2 layers (regular and BMQI)
  const lod2Layers = layers.generated || [];
  const visibleLod2Regular = lod2Layers.filter(layer =>
    visibleLayers.has(layer.id) && !layer.name.includes('BMQI')
  );
  const visibleLod2BMQI = lod2Layers.filter(layer =>
    visibleLayers.has(layer.id) && layer.name.includes('BMQI')
  );
  const hasVisibleLod2 = visibleLod2Regular.length > 0;
  const hasVisibleLod2BMQI = visibleLod2BMQI.length > 0;

  // Check if there are any visible LoD3 layers
  const lod3Layers = layers.lod3 || [];
  const hasVisibleLod3 = lod3Layers.some(layer => visibleLayers.has(layer.id));

  // Show LoD1 legend only when LoD1 is visible and no LoD2/LoD3 is visible
  const showLod1Legend = hasVisibleLod1 && !hasVisibleLod2 && !hasVisibleLod2BMQI && !hasVisibleLod3;

  // Show LoD2 legend only when LoD2 is visible and no LoD3 is visible
  const showLod2Legend = (hasVisibleLod2 || hasVisibleLod2BMQI) && !hasVisibleLod3;

  const customLayers = layers.custom || [];

  const LayerSection = ({ title, layers, showZoom, layerType }) => (
    <div className="layer-section">
      <div className="section-header">
        {title}
        {layerManager && (
          <span style={{ fontSize: '10px', color: '#666', marginLeft: '8px' }}>
            ({layers.length})
          </span>
        )}
      </div>
      {layers.map((layer) => {
        const isVisible = visibleLayers.has(layer.id);
        const layerInfo = layerManager?.getLayer(layer.id);
        
        return (
          <div key={layer.id} className="layer-item" onClick={() => handleLayerToggle(layer.id)} style={{ display: 'flex', alignItems: 'center' }}>
            <div className={`layer-checkbox ${isVisible ? 'checked' : ''}`} />
            <div className="layer-name" title={layerInfo?.metadata?.originalFile || layer.name}>
              {layer.name}
              {layerInfo?.geojsonId && (
                <span style={{ fontSize: '10px', color: '#8A2BE2', display: 'block' }}>
                  → from {layerManager.getLayer(layerInfo.geojsonId)?.name || 'GeoJSON'}
                </span>
              )}
            </div>
            {showZoom && (
              <button
                className="zoom-to-layer-btn"
                style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
                onClick={e => { 
                  e.stopPropagation(); 
                  handleZoomToLayer(layer.id, layerType || layer.type);
                }}
                title={`Zoom to ${layer.name}`}
              >
                <img src={zoomInIcon} alt="Zoom In" style={{ width: 20, height: 20 }} />
              </button>
            )}
          </div>
        );
      })}
      {layers.length === 0 && (
        <div style={{ fontSize: '12px', color: '#999', fontStyle: 'italic', padding: '8px 0' }}>
          No {title.toLowerCase()} layers
        </div>
      )}
    </div>
  );

  return (
    <div className="layer-panel">
      <LayerSection title="GEOJSON" layers={layers.geojson || []} showZoom={true} layerType="geojson" />
      <LayerSection title="LOD1モデル" layers={layers.lod1 || []} showZoom={true} layerType="lod1" />
      <LayerSection title="LOD2モデル" layers={layers.generated || []} showZoom={true} layerType="lod2" />
      <LayerSection title="LOD3モデル" layers={layers.lod3 || []} showZoom={true} layerType="lod3" />
      {customLayers.length > 0 && (
        <LayerSection title="衛星画像" layers={customLayers} showZoom={true} layerType="orthophoto" />
      )}
      
      {/* LoD1 Legend - shown only when LoD1 layers are visible and no LoD2 is visible */}
      {showLod1Legend && (
        <div className="layer-section">
          <div className="section-header">凡例</div>
          <div style={{ marginBottom: '12px' }}>
            <div className="legend-title">LoD1モデル</div>
            <div>
              {lod1Legend.map((item) => (
                <div key={item.label} className="legend-item">
                  <div
                    className="legend-color"
                    style={{ backgroundColor: item.color }}
                  />
                  <div className="legend-label">{item.label}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* LoD2 Legends - shown when LoD2 layers are visible and no LoD3 is visible */}
      {showLod2Legend && (
        <div className="layer-section">
          <div className="section-header">凡例</div>

          {/* LoD2 Regular Legend */}
          {hasVisibleLod2 && (
            <div style={{ marginBottom: '12px' }}>
              <div className="legend-title">LoD2モデル</div>
              <div>
                {lod2Legend.map((item) => (
                  item.isRoadGroup ? (
                    <div key={item.label}>
                      <div className="legend-item" style={{ fontWeight: '500' }}>
                        <div className="legend-label">{item.label}</div>
                      </div>
                      <div style={{ paddingLeft: '12px' }}>
                        {item.roads.map((road) => (
                          <div key={road.label} className="legend-item">
                            <div
                              className="legend-color"
                              style={{ backgroundColor: road.color }}
                            />
                            <div className="legend-label">{road.label}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ) : (
                    <div key={item.label} className="legend-item">
                      <div
                        className="legend-color"
                        style={{ backgroundColor: item.color }}
                      />
                      <div className="legend-label">{item.label}</div>
                    </div>
                  )
                ))}
              </div>
            </div>
          )}

          {/* LoD2 BMQI Legend - Gradient */}
          {hasVisibleLod2BMQI && (
            <div style={{ marginBottom: '12px' }}>
              <div className="legend-title">LoD2モデル - BMQIノイズ評価</div>
              <div style={{ padding: '4px 0' }}>
                {/* Gradient bar */}
                <div
                  style={{
                    height: '12px',
                    borderRadius: '2px',
                    background: lod2BMQIGradient.gradient,
                    marginBottom: '4px'
                  }}
                />
                {/* Labels */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  fontSize: '11px',
                  color: '#666'
                }}>
                  <span>{lod2BMQIGradient.lowLabel}</span>
                  <span>{lod2BMQIGradient.highLabel}</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* LoD3 Legend - shown when LoD3 layers are visible */}
      {hasVisibleLod3 && (
        <div className="layer-section">
          <div className="section-header">凡例</div>
          <div style={{ marginBottom: '12px' }}>
            <div className="legend-title">LOD3モデル</div>
            <div>
              {lod3Legend.map((item) => (
                item.isBuildingGroup ? (
                  <div key={item.label}>
                    <div className="legend-item" style={{ fontWeight: '500' }}>
                      <div className="legend-label">{item.label}</div>
                    </div>
                    <div style={{ paddingLeft: '12px' }}>
                      {item.parts.map((part) => (
                        <div key={part.label} className="legend-item">
                          <div
                            className="legend-color"
                            style={{ backgroundColor: part.color }}
                          />
                          <div className="legend-label">{part.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : item.isRoadGroup ? (
                  <div key={item.label}>
                    <div className="legend-item" style={{ fontWeight: '500' }}>
                      <div className="legend-label">{item.label}</div>
                    </div>
                    <div style={{ paddingLeft: '12px' }}>
                      {item.roads.map((road) => (
                        <div key={road.label} className="legend-item">
                          <div
                            className="legend-color"
                            style={{ backgroundColor: road.color }}
                          />
                          <div className="legend-label">{road.label}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div key={item.label} className="legend-item">
                    <div
                      className="legend-color"
                      style={{ backgroundColor: item.color }}
                    />
                    <div className="legend-label">{item.label}</div>
                  </div>
                )
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Debug information when layer manager is available */}
      {/* {layerManager && process.env.NODE_ENV === 'development' && (
        <div className="layer-section" style={{ fontSize: '10px', color: '#666' }}>
          <div className="section-header">DEBUG INFO</div>
          <div>Total layers: {layerManager.getAllLayers().length}</div>
          <div>Visible: {Array.from(visibleLayers).length}</div>
        </div>
      )} */}
    </div>
  );
};

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Compare object/array props by reference
  if (prevProps.layers !== nextProps.layers) return false;
  if (prevProps.visibleLayers !== nextProps.visibleLayers) return false;
  if (prevProps.layerManager !== nextProps.layerManager) return false;

  // Compare callbacks (should be memoized with useCallback in parent)
  if (prevProps.onLayerToggle !== nextProps.onLayerToggle) return false;
  if (prevProps.zoomToLayer !== nextProps.zoomToLayer) return false;
  if (prevProps.zoomToOrthophoto !== nextProps.zoomToOrthophoto) return false;

  return true;
};

export default React.memo(LayerPanel, arePropsEqual);