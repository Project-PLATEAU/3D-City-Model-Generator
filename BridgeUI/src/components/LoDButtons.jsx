import React, { useState } from 'react';
import '../styles/Components.css';

// Import SVG icons
import LoD1Icon from '../assets/icons/LoD1.svg';
import LoD2Icon from '../assets/icons/LoD2.svg';
import LoD3Icon from '../assets/icons/LoD3.svg';

const LoDButtons = ({ projectStage, availableLoDs, onLoDSelect }) => {
  const [selectedLoD, setSelectedLoD] = useState(null);

  const handleLoDClick = (lodLevel) => {
    if (isLoDAvailable(lodLevel)) {
      setSelectedLoD(lodLevel);
      onLoDSelect && onLoDSelect(lodLevel);
    }
  };

  const isLoDAvailable = (lodLevel) => {
    switch (lodLevel) {
      case 'lod1':
        return projectStage === 'geojson-imported' || availableLoDs.has('lod1');
      case 'lod2':
        return availableLoDs.has('lod1') && availableLoDs.has('lod2');
      case 'lod3':
        return availableLoDs.has('lod2') && availableLoDs.has('lod3');
      default:
        return false;
    }
  };

  const getLoDIcon = (lodLevel) => {
    const icons = {
      lod1: LoD1Icon,
      lod2: LoD2Icon,
      lod3: LoD3Icon
    };
    return icons[lodLevel] || LoD1Icon;
  };

  return (
    <div className="lod-buttons-container">
      <div className="lod-buttons">
        {['lod1', 'lod2', 'lod3'].map((lodLevel) => {
          const isAvailable = isLoDAvailable(lodLevel);
          const isSelected = selectedLoD === lodLevel;
          
          return (
            <button
              key={lodLevel}
              className={`lod-button ${isAvailable ? 'available' : 'frozen'} ${isSelected ? 'selected' : ''}`}
              onClick={() => handleLoDClick(lodLevel)}
              disabled={!isAvailable}
              title={isAvailable ? `Configure ${lodLevel.toUpperCase()}` : `${lodLevel.toUpperCase()} - Prerequisites not met`}
            >
              <div className="lod-icon">
                <img 
                  src={getLoDIcon(lodLevel)} 
                  alt={`${lodLevel.toUpperCase()} icon`}
                  className="lod-svg-icon"
                />
              </div>
              <div className="lod-label">
                {lodLevel.toUpperCase()}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Compare primitive props
  if (prevProps.projectStage !== nextProps.projectStage) return false;

  // Compare object props by reference
  if (prevProps.availableLoDs !== nextProps.availableLoDs) return false;

  // Compare callbacks
  if (prevProps.onLoDSelect !== nextProps.onLoDSelect) return false;

  return true;
};

export default React.memo(LoDButtons, arePropsEqual);
