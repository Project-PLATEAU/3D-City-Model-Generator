# Bridge UI Frontend Development
The platform is built to visualize the generation process and results of 3D city models at different LoDs from orthophoto and height input (optionally point cloud). The result can then be exported as .obj/.gml.

## Implementation
- UI components: ooooooooo- 90%
- base map: mapbox gl (token currently under the author's UTokyo email account): oooooooooo done
- 3D view functions: ooooooooo-  90%
- inspecting each building: ----------- not started yet

# UI Design
### Main Page
![Bridge UI Screenshot 1](README_img/Bridge%20UI_1.jpg)

### Model Generation
Data Import

![Bridge UI Screenshot 2](README_img/Bridge%20UI_2.jpg)
![Bridge UI Screenshot 3](README_img/Bridge%20UI_3.jpg)

Choice of LoD

![Bridge UI Screenshot 4](README_img/Bridge%20UI_4.jpg)

Processing Page

![Bridge UI Screenshot 5](README_img/Bridge%20UI_5.jpg)
![Bridge UI Screenshot 6](README_img/Bridge%20UI_6.jpg)

### After Import
![Bridge UI Screenshot 7](README_img/Bridge%20UI_7.jpg)

### Extra Information
![Bridge UI Screenshot 8](README_img/Bridge%20UI_8.jpg)

# Tech-stack
This frontend is based on Vite with React template.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh
