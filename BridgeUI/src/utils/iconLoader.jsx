// Icon loader utility
import handIcon from '../assets/icons/hand.svg?url';
import zoomInIcon from '../assets/icons/zoom-in.svg?url';
import zoomOutIcon from '../assets/icons/zoom-out.svg?url';
import layersIcon from '../assets/icons/layers.svg?url';
//import mapIcon from '../assets/icons/map.svg?url';
import uploadIcon from '../assets/icons/upload.svg?url';
import geojsonIcon from '../assets/icons/geojson.svg?url';
import orthophotoIcon from '../assets/icons/orthophoto.svg?url';
import pointcloudIcon from '../assets/icons/pointcloud.svg?url';
import streetViewIcon from '../assets/icons/streetView.svg?url';
import LoD1Icon from '../assets/icons/LoD1.svg?url';
import LoD2Icon from '../assets/icons/LoD2.svg?url';
import LoD3Icon from '../assets/icons/LoD3.svg?url';

export const icons = {
  hand: handIcon,
  zoomIn: zoomInIcon,
  zoomOut: zoomOutIcon,
  layers: layersIcon,
  //map: mapIcon,
  upload: uploadIcon,
  geojson: geojsonIcon,
  orthophoto: orthophotoIcon,
  pointcloud: pointcloudIcon,
  streetView: streetViewIcon,
  LoD1: LoD1Icon,
  LoD2: LoD2Icon,
  LoD3: LoD3Icon,
};

// Helper component for rendering icons
export const Icon = ({ name, size = 20, className = '', style = {}, noFilter = false }) => {
  const iconSrc = icons[name];
  
  if (!iconSrc) {
    console.warn(`Icon "${name}" not found`);
    return null;
  }
  
  const defaultStyle = noFilter ? {} : { filter: 'brightness(0) invert(1)' }; // Makes icons white by default
  
  return (
    <img 
      src={iconSrc} 
      alt={name}
      width={size}
      height={size}
      className={className}
      style={{ ...defaultStyle, ...style }}
    />
  );
};