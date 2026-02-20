import { useState, useRef, useEffect } from 'react';
import '../styles/Components.css';
import { Icon } from '../utils/iconLoader';

const ImportModel = ({ isOpen, onClose, onImport, projectStage = 'initial', availableLoDs = new Set() }) => {
  const [layerName, setLayerName] = useState('');
  const [progress, setProgress] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState(null);
  const [fileStatus, setFileStatus] = useState('idle'); // 'idle' | 'loading' | 'success' | 'error'
  const [fileError, setFileError] = useState('');
  const [previewUrl, setPreviewUrl] = useState(null);
  const [selectedType, setSelectedType] = useState(null); // null | 'geojson' | 'orthophoto' | 'pointcloud'
  const [showTypeSelection, setShowTypeSelection] = useState(true);
  const [showLoDSelection, setShowLoDSelection] = useState(false); // New state for LoD selection
  const [selectedLoD, setSelectedLoD] = useState(null); // Track selected LoD
  const [lodConfig, setLoDConfig] = useState({ height: '', roofType: 'flat' }); // LoD configuration
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (previewUrl) {
      return () => {
        URL.revokeObjectURL(previewUrl);
      };
    }
  }, [previewUrl]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleKeyDown);
    }

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [isOpen, onClose]);

  // Reset state when modal opens
  useEffect(() => {
    if (isOpen) {
      setSelectedType(null);
      setShowTypeSelection(true);
      setShowLoDSelection(false);
      setSelectedLoD(null);
      setLoDConfig({ height: '', roofType: 'flat' });
      setLayerName('');
      setFile(null);
      setFileStatus('idle');
      setFileError('');
      setPreviewUrl(null);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleFileUpload = (files) => {
    if (files.length > 0) {
      setFileStatus('loading');
      setFileError('');
      setFile(null);
      setPreviewUrl(null);
      const selectedFile = files[0];
      
      // Simulate file validation/loading
      setTimeout(() => {
        // Validate file based on selected type
        const isValidFile = validateFileType(selectedFile, selectedType);
        
        if (!isValidFile) {
          setFileStatus('error');
          setFileError(getFileTypeErrorMessage(selectedType));
          return;
        }
        
        setFile(selectedFile);
        setFileStatus('success');
        setFileError('');
        
        // Create preview for image files
        if (selectedFile.type.startsWith('image/')) {
          setPreviewUrl(URL.createObjectURL(selectedFile));
        }
      }, 1000); // Simulate loading delay
    }
  };

  const validateFileType = (file, type) => {
    const fileName = file.name.toLowerCase();
    const fileType = file.type.toLowerCase();
    
    switch (type) {
      case 'geojson':
        return fileName.endsWith('.geojson') || fileName.endsWith('.json') || fileType === 'application/json';
      case 'orthophoto':
        return fileType.startsWith('image/') || fileName.match(/\.(jpg|jpeg|png|tiff|tif)$/);
      case 'pointcloud':
        return fileName.match(/\.(ply|las|laz|pcd|xyz)$/);
      default:
        return false;
    }
  };

  const getFileTypeErrorMessage = (type) => {
    switch (type) {
      case 'geojson':
        return 'Please select a valid GeoJSON file (.geojson, .json).';
      case 'orthophoto':
        return 'Please select a valid image file (.jpg, .png, .tiff).';
      case 'pointcloud':
        return 'Please select a valid point cloud file (.ply, .las, .pcd).';
      default:
        return 'Invalid file type.';
    }
  };

  const getAcceptedFileTypes = (type) => {
    switch (type) {
      case 'geojson':
        return '.geojson,.json,application/json';
      case 'orthophoto':
        return 'image/*,.tiff,.tif';
      case 'pointcloud':
        return '.ply,.las,.laz,.pcd,.xyz';
      default:
        return '*';
    }
  };

  const handleTypeSelect = (type) => {
    setSelectedType(type);
    setShowTypeSelection(false);
  };

  const handleBack = () => {
    if (showLoDSelection) {
      setShowLoDSelection(false);
      setShowTypeSelection(false);
      setSelectedLoD(null);
      setLoDConfig({ height: '', roofType: 'flat' });
    } else {
      setShowTypeSelection(true);
      setSelectedType(null);
      setFile(null);
      setFileStatus('idle');
      setFileError('');
      setPreviewUrl(null);
      setLayerName('');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const files = Array.from(e.dataTransfer.files);
    handleFileUpload(files);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setDragOver(false);
  };

  const handleNext = () => {
    if (showLoDSelection && selectedLoD && lodConfig.height) {
      // Generate LoD data
      onImport && onImport({ 
        type: 'lod-generation',
        lodType: selectedLoD,
        config: lodConfig,
        sourceType: selectedType
      });
    } else if (fileStatus === 'success' && layerName && file) {
      // Import file data
      onImport && onImport({ file, layerName, type: selectedType, previewUrl });
    }
  };

  const renderTypeSelection = () => {
    const fileTypes = [
      {
        id: 'pointcloud',
        name: 'Point Cloud',
        description: 'LoD3 Generation',
        iconName: 'pointcloud',
        lodIconName: 'LoD3',
        formats: '.ply, .las, .pcd',
        enabled: projectStage === 'orthophoto-imported' || availableLoDs.has('lod2'),
        disabledReason: projectStage !== 'orthophoto-imported' && !availableLoDs.has('lod2') ? 'Import orthophoto first' : null
      },
      {
        id: 'orthophoto',
        name: 'Orthophoto',
        description: 'LoD2 Generation',
        iconName: 'orthophoto',
        lodIconName: 'LoD2',
        formats: '.tiff',
        enabled: availableLoDs.has('lod1'),
        disabledReason: !availableLoDs.has('lod1') ? 'Generate LoD1 data first' : null
      },
      {
        id: 'geojson',
        name: 'GeoJSON',
        description: 'LoD1 Generation',
        iconName: 'geojson',
        lodIconName: 'LoD1',
        formats: '.geojson',
        enabled: projectStage === 'initial',
        disabledReason: projectStage !== 'initial' ? 'GeoJSON already imported' : null
      }
    ];

    return (
      <div style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'flex-start', 
        flex: 1, 
        padding: '10px 0 12px 0', 
        width: '100%',
        minHeight: 0, // Allow shrinking
        overflow: 'hidden' // Prevent overflow from this container
      }}>
        {/* Scrollable container for the buttons */}
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          width: '100%',
          flex: 1,
          minHeight: 0, // Allow shrinking
          overflow: 'auto', // Enable scrolling
          paddingBottom: '20px', // Space before the fixed text
        }}>
          <div style={{
            display: 'flex',
            flexDirection: 'row',
            flexWrap: 'wrap',
            gap: '20px',
            width: '100%',
            maxWidth: '480px',
            justifyContent: 'center',
            alignItems: 'center',
            padding: '0 20px', // Add horizontal padding for scrolling
            boxSizing: 'border-box'
          }}>
            {fileTypes.map((type) => (
              <div
                key={type.id}
                onClick={() => type.enabled && handleTypeSelect(type.id)}
                className="file-type-card"
                style={{
                  border: type.enabled ? '2px solid #e0e0e0' : '2px solid #ccc',
                  borderRadius: '12px',
                  padding: '18px 24px',
                  cursor: type.enabled ? 'pointer' : 'not-allowed',
                  textAlign: 'left',
                  transition: 'all 0.3s ease',
                  backgroundColor: type.enabled ? '#fff' : '#f5f5f5',
                  opacity: type.enabled ? 1 : 0.6,
                  display: 'flex',
                  alignItems: 'center',
                  minHeight: '120px',
                  boxShadow: type.enabled ? '0 2px 8px rgba(140,43,226,0.04)' : 'none',
                  width: '100%',
                  maxWidth: '440px',
                  flex: '1 1 320px',
                  margin: 0,
                  flexShrink: 0 // Prevent buttons from shrinking
                }}
              >
                {/* Icon group on the left */}
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minWidth: 120,
                  marginRight: 40,
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                    <Icon
                      name={type.iconName}
                      size={64}
                      noFilter={!type.enabled}
                      style={{ filter: type.enabled ? 'none' : 'grayscale(100%)', stroke: (['LoD1','LoD2','LoD3'].includes(type.lodIconName) ? '#7b6ae9' : undefined) }}
                    />
                    <span style={{ fontSize: 32, color: type.enabled ? '#8A2BE2' : '#ccc', fontWeight: 'bold', margin: '0 8px' }}>→</span>
                    <Icon
                      name={type.lodIconName}
                      size={64}
                      noFilter={!type.enabled}
                      style={{ filter: type.enabled ? 'none' : 'grayscale(100%)', stroke: (['LoD1','LoD2','LoD3'].includes(type.lodIconName) ? '#7b6ae9' : undefined) }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginTop: 6 }}>
                    <div style={{ fontSize: 12, color: type.enabled ? '#666' : '#999', fontWeight: 500, textAlign: 'center', width: 64 }}>Source</div>
                    <div style={{ fontSize: 12, color: type.enabled ? '#666' : '#999', fontWeight: 500, textAlign: 'center', width: 64 }}>{type.lodIconName}</div>
                  </div>
                </div>
                {/* Text group on the right */}
                <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: 4, color: type.enabled ? '#333' : '#999' }}>
                    {type.name}
                  </div>
                  <div style={{ fontSize: '14px', color: type.enabled ? '#666' : '#999', marginBottom: 4, lineHeight: '1.4' }}>
                    {type.description}
                  </div>
                  <div style={{ fontSize: '12px', color: type.enabled ? '#888' : '#aaa', fontStyle: 'italic', marginBottom: 2 }}>
                    Supported: {type.formats}
                  </div>
                  {!type.enabled && type.disabledReason && (
                    <div style={{ fontSize: '12px', color: '#8A2BE2', marginTop: '4px', fontWeight: '500' }}>
                      {type.disabledReason}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
        
        {/* Fixed instruction text at the bottom */}
        <div style={{ 
          fontSize: '18px', 
          color: '#333', 
          textAlign: 'center', 
          width: '100%',
          padding: '16px 20px 0 20px',
          borderTop: '1px solid #f0f0f0', // Subtle separator
          flexShrink: 0 // Prevent this text from being compressed
        }}>
          Select the data type to import
        </div>
      </div>
    );
  };

  const renderLoDSelection = () => {
    const lodTypes = [
      {
        id: 'lod1',
        name: 'LoD 1',
        description: 'Block model with basic geometric representation',
        iconName: 'LoD1',
        enabled: selectedType === 'geojson', // Only LoD1 for GeoJSON
        configFields: ['height']
      },
      {
        id: 'lod2',
        name: 'LoD 2',
        description: 'Detailed model with roof structures',
        iconName: 'LoD2',
        enabled: selectedType === 'orthophoto', // Only LoD2 for orthophoto
        configFields: ['height', 'roofType']
      },
      {
        id: 'lod3',
        name: 'LoD 3',
        description: 'Highly detailed model with architectural features',
        iconName: 'LoD3',
        enabled: selectedType === 'pointcloud', // Only LoD3 for point cloud
        configFields: ['height', 'roofType', 'wallTexture']
      }
    ];

    return (
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'stretch', justifyContent: 'flex-start', flex: 1, padding: '20px 0', width: '100%' }}>
        <div style={{ fontSize: '18px', marginBottom: '32px', color: '#333', textAlign: 'left' }}>
          Select Level of Detail (LoD) to generate:
        </div>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '20px', width: '100%' }}>
          {lodTypes.map((lod) => (
            <div
              key={lod.id}
              onClick={() => lod.enabled && setSelectedLoD(lod.id)}
              className="file-type-card"
              style={{
                border: selectedLoD === lod.id ? '3px solid #8A2BE2' : (lod.enabled ? '2px solid #e0e0e0' : '2px solid #ccc'),
                borderRadius: '12px',
                padding: '24px',
                cursor: lod.enabled ? 'pointer' : 'not-allowed',
                textAlign: 'center',
                transition: 'all 0.3s ease',
                backgroundColor: selectedLoD === lod.id ? '#f8f4ff' : (lod.enabled ? '#fff' : '#f5f5f5'),
                opacity: lod.enabled ? 1 : 0.6
              }}
            >
              <div style={{ marginBottom: '16px', display: 'flex', justifyContent: 'center' }}>
                <Icon 
                  name={lod.iconName} 
                  size={96} 
                  noFilter={lod.enabled}
                  style={{ filter: lod.enabled ? 'none' : 'grayscale(100%)', stroke: (['LoD1','LoD2','LoD3'].includes(lod.iconName) ? '#7b6ae9' : undefined) }}
                />
              </div>
              <div style={{ fontSize: '20px', fontWeight: 'bold', marginBottom: '8px', color: lod.enabled ? '#333' : '#999' }}>
                {lod.name}
              </div>
              <div style={{ fontSize: '14px', color: lod.enabled ? '#666' : '#999', marginBottom: '12px', lineHeight: '1.4' }}>
                {lod.description}
              </div>
              {!lod.enabled && (
                <div style={{ fontSize: '12px', color: '#8A2BE2', marginTop: '8px', fontWeight: '500' }}>
                  {selectedType === 'geojson' && lod.id !== 'lod1' ? 'Only LoD1 available for GeoJSON' :
                   selectedType === 'orthophoto' && lod.id !== 'lod2' ? 'Only LoD2 available for Orthophoto' :
                   selectedType === 'pointcloud' && lod.id !== 'lod3' ? 'Only LoD3 available for Point Cloud' :
                   'Not available for this data type'}
                </div>
              )}
            </div>
          ))}
        </div>
        
        {selectedLoD && (
          <div style={{ marginTop: '32px', padding: '20px', border: '1px solid #e0e0e0', borderRadius: '8px', width: '100%' }}>
            <div style={{ fontSize: '16px', fontWeight: 'bold', marginBottom: '16px', color: '#333' }}>
              Configuration for {selectedLoD.toUpperCase()}
            </div>
            
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#555' }}>
                Height (meters):
              </label>
              <input
                type="number"
                value={lodConfig.height}
                onChange={(e) => setLoDConfig(prev => ({ ...prev, height: e.target.value }))}
                placeholder="Enter building height"
                style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
              />
            </div>
            
            {(selectedLoD === 'lod2' || selectedLoD === 'lod3') && (
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500', color: '#555' }}>
                  Roof Type:
                </label>
                <select
                  value={lodConfig.roofType}
                  onChange={(e) => setLoDConfig(prev => ({ ...prev, roofType: e.target.value }))}
                  style={{ width: '100%', padding: '8px', border: '1px solid #ddd', borderRadius: '4px' }}
                >
                  <option value="flat">Flat Roof</option>
                  <option value="gable">Gable Roof</option>
                  <option value="hip">Hip Roof</option>
                  <option value="shed">Shed Roof</option>
                </select>
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  const renderPreviewSection = () => {
    if (!selectedType) return null;

    const getTypeDisplayName = (type) => {
      switch (type) {
        case 'geojson': return 'GeoJSON';
        case 'orthophoto': return 'Orthophoto';
        case 'pointcloud': return 'Point Cloud';
        default: return 'File';
      }
    };

    const getPlaceholderText = (type) => {
      switch (type) {
        case 'geojson': return 'Drop GeoJSON file here or click to browse';
        case 'orthophoto': return 'Drop orthophoto file (GeoTIFF) here or click to browse';
        case 'pointcloud': return 'Drop point cloud file here or click to browse';
        default: return 'Drop file here or click to browse';
      }
    };

    return (
      <div className="preview-container" style={{ flex: '1 1 0', display: 'flex', flexDirection: 'column', justifyContent: 'flex-start', alignItems: 'stretch', marginBottom: '0', minHeight: '200px', width: '100%' }}>
        <div className="preview-box single" style={{ position: 'relative', flex: '1 1 0', minHeight: '200px', width: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', border: '2px dashed #cccccc', borderRadius: '8px', background: 'rgba(255,255,255,0.5)', boxSizing: 'border-box' }}>
          {previewUrl && selectedType === 'orthophoto' ? (
            <img src={previewUrl} alt="Orthophoto preview" style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', objectFit: 'contain', borderRadius: '6px', zIndex: 1 }} />
          ) : null}
          {!previewUrl && (
            <>
              <div className="preview-title" style={{ fontWeight: 'bold', marginBottom: '16px', textAlign: 'center', zIndex: 2 }}>
                Preview {getTypeDisplayName(selectedType)}
              </div>
              <div 
                className="preview-placeholder"
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
                style={{
                  border: dragOver ? '2px dashed #8A2BE2' : '2px dashed #cccccc',
                  backgroundColor: dragOver ? 'rgba(138, 43, 226, 0.1)' : 'transparent',
                  width: '320px',
                  height: '40px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto',
                  fontSize: '16px',
                  color: '#666',
                  marginBottom: '0',
                  cursor: 'pointer',
                  zIndex: 2,
                }}
              >
                {fileStatus === 'loading' && (
                  <div className="preview-status loading">Loading...</div>
                )}
                {fileStatus === 'success' && file && (
                  <div className="preview-status success">File loaded: {file.name}</div>
                )}
                {fileStatus === 'error' && fileError && (
                  <div className="preview-status error">{fileError}</div>
                )}
                {fileStatus === 'idle' && !dragOver && (
                  <div className="preview-placeholder-text">{getPlaceholderText(selectedType)}</div>
                )}
                <input
                  type="file"
                  accept={getAcceptedFileTypes(selectedType)}
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  onChange={e => handleFileUpload(Array.from(e.target.files))}
                />
              </div>
            </>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="modal-overlay">
      <div
        className="import-modal"
        style={showTypeSelection
          ? {
              display: 'flex', flexDirection: 'column', maxHeight: '90vh', height: 'auto', minHeight: '0', width: '100%', maxWidth: '520px', padding: '32px 0 24px 0', boxSizing: 'border-box', justifyContent: 'flex-start', alignItems: 'center', margin: '0 auto'
            }
          : {
              display: 'flex', flexDirection: 'column', maxHeight: '80vh', height: '100%', padding: '40px', boxSizing: 'border-box', justifyContent: 'flex-start', alignItems: 'stretch', margin: '0 auto', width: '800px', minWidth: '600px'
            }
        }
      >
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '24px', width: '100%' }}>
          {!showTypeSelection && (
            <button
              onClick={handleBack}
              style={{
                background: 'none',
                border: 'none',
                fontSize: '18px',
                cursor: 'pointer',
                marginRight: '16px',
                color: '#8A2BE2',
                display: 'flex',
                alignItems: 'center'
              }}
            >
              ← Back
            </button>
          )}
          <div className="import-modal-title" style={{ margin: '0 0 0 36px' }}>
            {showTypeSelection ? 'IMPORT SOURCE DATA' : 
             showLoDSelection ? `CONFIGURE ${selectedType?.toUpperCase()} LoD` :
             `IMPORT ${selectedType?.toUpperCase()}`}
          </div>
        </div>
        
        {showTypeSelection ? renderTypeSelection() : 
         showLoDSelection ? (
          <>
            {renderLoDSelection()}
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '24px', gap: '16px' }}>
              <button
                className="model-button primary"
                disabled={!selectedLoD || !lodConfig.height}
                style={{ 
                  minWidth: '120px', 
                  height: '40px', 
                  fontSize: '22px', 
                  background: (!selectedLoD || !lodConfig.height) ? '#ccc' : '#8BC34A', 
                  color: '#fff', 
                  borderRadius: '8px', 
                  fontWeight: 'bold', 
                  border: 'none', 
                  cursor: (!selectedLoD || !lodConfig.height) ? 'not-allowed' : 'pointer' 
                }}
                onClick={handleNext}
              >
                GENERATE {selectedLoD?.toUpperCase()}
              </button>
            </div>
          </>
         ) : (
          <>
            {renderPreviewSection()}
            <div style={{ display: 'flex', alignItems: 'center', width: '100%', marginTop: '24px', marginBottom: '8px', gap: '16px' }}>
              <div style={{ flex: 2, display: 'flex', alignItems: 'center' }}>
                <label style={{ fontWeight: '500', fontSize: '16px', color: '#222', marginRight: '8px', whiteSpace: 'nowrap' }}>
                  {selectedType === 'geojson' ? 'GeoJSON' : selectedType === 'orthophoto' ? 'Orthophoto' : 'Point Cloud'} Layer Name
                </label>
                <input
                  type="text"
                  className="import-input"
                  value={layerName}
                  onChange={(e) => setLayerName(e.target.value)}
                  placeholder=""
                  style={{ width: '180px', height: '28px', fontSize: '16px', marginRight: '8px' }}
                />
              </div>
              <div style={{ flex: 1, textAlign: 'center', color: '#222', fontSize: '16px', minHeight: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                {fileStatus === 'idle' && 'Waiting for Input'}
                {fileStatus === 'loading' && 'Loading...'}
                {fileStatus === 'success' && 'Completed!'}
                {fileStatus === 'error' && fileError}
              </div>
              <div style={{ flex: 1, display: 'flex', justifyContent: 'flex-end' }}>
                <button
                  className="model-button primary"
                  disabled={!layerName || fileStatus !== 'success'}
                  style={{ minWidth: '120px', height: '40px', fontSize: '22px', background: '#8BC34A', color: '#fff', borderRadius: '8px', fontWeight: 'bold', border: 'none', cursor: (!layerName || fileStatus !== 'success') ? 'not-allowed' : 'pointer' }}
                  onClick={handleNext}
                >
                  NEXT
                </button>
              </div>
            </div>
            <div style={{ width: '100%', height: '8px', background: '#eee', borderRadius: '4px', marginTop: '0', marginBottom: '0', overflow: 'hidden' }}>
              <div style={{ width: fileStatus === 'loading' ? '50%' : fileStatus === 'success' ? '100%' : '0%', height: '100%', background: fileStatus === 'success' ? '#8BC34A' : '#8A2BE2', transition: 'width 0.5s' }} />
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default ImportModel;