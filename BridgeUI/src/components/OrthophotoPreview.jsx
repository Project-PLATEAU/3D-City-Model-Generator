import React from 'react';

const OrthophotoPreview = ({ mapInstance, orthophotoData }) => {
  // Use the orthophotoData prop directly as the preview image
  const previewImage = orthophotoData;

  return (
    <div className="orthophoto-panel">
      <div className="orthophoto-content">
        <div className="orthophoto-header">衛星画像</div>
        <div className="orthophoto-preview">
          {previewImage ? (
            <img 
              src={previewImage} 
              alt="Orthophoto Preview" 
              style={{ width: '100%', height: '100%', objectFit: 'cover' }}
            />
          ) : (
            <div className="orthophoto-placeholder">
              NO DATA
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// Custom comparison function to prevent unnecessary re-renders
const arePropsEqual = (prevProps, nextProps) => {
  // Compare object props by reference
  if (prevProps.mapInstance !== nextProps.mapInstance) return false;
  if (prevProps.orthophotoData !== nextProps.orthophotoData) return false;

  return true;
};

export default React.memo(OrthophotoPreview, arePropsEqual);