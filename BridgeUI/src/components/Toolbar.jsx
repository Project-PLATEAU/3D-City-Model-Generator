import React from 'react';
import { Icon } from '../utils/iconLoader.jsx';

const Toolbar = ({ onToolChange, activeTool, onImportClick }) => {
  const mainTools = [
    { id: 'hand', icon: 'hand', tooltip: 'Pan Tool' },
    // { id: 'zoom-in', icon: 'zoomIn', tooltip: 'Zoom In' },
    // { id: 'zoom-out', icon: 'zoomOut', tooltip: 'Zoom Out' },
    { id: 'layers', icon: 'layers', tooltip: 'Layer Stack' },
    //{ id: 'map', icon: 'map', tooltip: 'Map' }
  ];

  const handleToolClick = (toolId) => {
    if (toolId === 'import') {
      onImportClick && onImportClick();
    } else {
      onToolChange && onToolChange(toolId);
    }
  };

  return (
    <div className="toolbar">
      <div className="toolbar-main">
        {mainTools.map((tool) => (
          <button
            key={tool.id}
            className={`toolbar-button ${activeTool === tool.id ? 'active' : ''}`}
            onClick={() => handleToolClick(tool.id)}
            title={tool.tooltip}
          >
            <Icon name={tool.icon} size={32} />
          </button>
        ))}
      </div>
      <div className="toolbar-import">
        <button
          className="toolbar-button import-button"
          onClick={() => handleToolClick('import')}
          title="Import Data"
        >
          <Icon name="upload" size={32} />
          <span className="import-text">インポート</span>
        </button>
      </div>
    </div>
  );
};

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Compare primitive props
  if (prevProps.activeTool !== nextProps.activeTool) return false;

  // Compare callbacks (should be memoized with useCallback in parent)
  if (prevProps.onToolChange !== nextProps.onToolChange) return false;
  if (prevProps.onImportClick !== nextProps.onImportClick) return false;

  return true;
};

export default React.memo(Toolbar, arePropsEqual);