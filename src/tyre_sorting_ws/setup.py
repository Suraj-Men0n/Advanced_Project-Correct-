from setuptools import setup, find_packages
import glob
package_name = 'tyre_sorting'
setup(name=package_name, version='0.2.0', packages=find_packages('src'), package_dir={'':'src'},
      data_files=[('share/ament_index/resource_index/packages', ['resource/tyre_sorting']),
                  ('share/tyre_sorting', ['package.xml', 'launch/full_system.launch.py']),
                  (f'share/{package_name}/dataset/meshes', glob.glob('dataset/meshes/*.obj')),
                  (f'share/{package_name}/dataset', ['dataset/metadata.csv', 'dataset/spectra.npz']),
                  (f'share/{package_name}/models', ['models/spectral_cnn.pt', 'models/training_metrics.json', 'models/training_history.json'])],
      install_requires=['setuptools'], zip_safe=True,
      entry_points={'console_scripts': [
          'generate_dataset = tyre_sorting.data.generate_dataset:build_dataset_cli',
          'train_spectral_cnn = tyre_sorting.perception.cnn_decoder:main',
          'routing_node = tyre_sorting.ros2_nodes.routing_node:main',
          'multimodal_fusion = tyre_sorting.ros2_nodes.fusion_node:main',
          'integration_runner = tyre_sorting.integration.closed_loop:main',
          'simulation_bridge = tyre_sorting.ros2_nodes.sim_bridge:main',
          'geometry_segmentation = tyre_sorting.perception.segmentation_node:main',
          'spectral_classifier = tyre_sorting.ros2_nodes.spectral_node:main',
          'pbvs_controller = tyre_sorting.control.pbvs_controller:main',
          'parametric_eval = tyre_sorting.eval.parametric_eval:main',
      ]})
