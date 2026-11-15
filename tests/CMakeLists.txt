set(COLITE_CORE_TEST_SOURCES
  controllers/sfm_camera_setup_test.cpp
  core/camera_test.cpp
  core/geometry_test.cpp
  core/match_graph_test.cpp
  core/parallel_test.cpp
  core/reconstruction_test.cpp
  features/sift_extractor_test.cpp
  features/feature_mask_test.cpp
  features/aliked_extractor_test.cpp
  features/official_onnx_golden_test.cpp
  features/sift_cpu_cuda_overlap_test.cpp
  integration/recon_mvs_exe_test.cpp
  integration/recon_sfm_exe_test.cpp
  integration/sfm_workspace_and_commands_test.cpp
  integration/sfm_dependency_boundary_test.cpp
  integration/stage_test.cpp
  io/image_reader_test.cpp
  io/model_resource_test.cpp
  matching/descriptor_matcher_test.cpp
  matching/learned_matcher_test.cpp
  matching/guided_matching_test.cpp
  matching/match_graph_consistency_test.cpp
  matching/pair_generator_test.cpp
  matching/two_view_geometry_batch_test.cpp
  matching/two_view_geometry_verifier_test.cpp
  matching/weak_pair_rematcher_test.cpp
  matching/sift_matching_pipeline_test.cpp
  mvs/dense_cloud_test.cpp
  mvs/dense_map_test.cpp
  mvs/image_workspace_test.cpp
  mvs/image_undistorter_test.cpp
  mvs/mvs_pipeline_test.cpp
  mvs/meshing_test.cpp
  mvs/mesh_texturer_test.cpp
  mvs/patch_match_test.cpp
  sfm/global_bundle_adjuster_test.cpp
  sfm/refinement_diagnostics_test.cpp
  sfm/global_mapper_test.cpp
  sfm/global_pose_graph_test.cpp
  sfm/hierarchical_mapper_test.cpp
  sfm/image_registrar_test.cpp
  sfm/incremental_initializer_test.cpp
  sfm/incremental_mapper_test.cpp
  sfm/incremental_triangulator_test.cpp
  sfm/local_bundle_adjuster_test.cpp
  sfm/model_clusterer_test.cpp
  sfm/observation_manager_test.cpp
  sfm/observation_filter_test.cpp
  sfm/reconstruction_manager_test.cpp
  sfm/similarity_transform_test.cpp
  sfm/structureless_registrar_test.cpp
)

set(COLITE_VLFEAT_REFERENCE_DIR
    "${PROJECT_SOURCE_DIR}/references/vlfeat-0.9.21")
set(COLITE_VLFEAT_REFERENCE_TARGET "")
if(EXISTS "${COLITE_VLFEAT_REFERENCE_DIR}/vl/sift.c")
  list(APPEND COLITE_CORE_TEST_SOURCES
    features/sift_reference_consistency_test.cpp
  )

  # The unmodified upstream Windows reference initializes its global state in
  # DllMain, so it must remain a DLL on Windows.
  set(COLITE_VLFEAT_REFERENCE_KIND OBJECT)
  if(WIN32)
    set(COLITE_VLFEAT_REFERENCE_KIND SHARED)
  endif()
  add_library(vlfeat_original_reference ${COLITE_VLFEAT_REFERENCE_KIND}
    ${COLITE_VLFEAT_REFERENCE_DIR}/vl/generic.c
    ${COLITE_VLFEAT_REFERENCE_DIR}/vl/host.c
    ${COLITE_VLFEAT_REFERENCE_DIR}/vl/imopv.c
    ${COLITE_VLFEAT_REFERENCE_DIR}/vl/random.c
    ${COLITE_VLFEAT_REFERENCE_DIR}/vl/sift.c
  )

  target_include_directories(vlfeat_original_reference
    PRIVATE
      ${COLITE_VLFEAT_REFERENCE_DIR}
  )

  target_compile_definitions(vlfeat_original_reference
    PRIVATE
      $<$<C_COMPILER_ID:MSVC>:VL_BUILD_DLL>
      VL_DISABLE_THREADS
      VL_DISABLE_SSE2
      VL_DISABLE_OPENMP
      VlSiftFilt=OrigVlSiftFilt
      VlSiftKeypoint=OrigVlSiftKeypoint
      vl_sift_new=orig_vl_sift_new
      vl_sift_delete=orig_vl_sift_delete
      vl_sift_process_first_octave=orig_vl_sift_process_first_octave
      vl_sift_process_next_octave=orig_vl_sift_process_next_octave
      vl_sift_detect=orig_vl_sift_detect
      vl_sift_calc_keypoint_orientations=orig_vl_sift_calc_keypoint_orientations
      vl_sift_calc_keypoint_descriptor=orig_vl_sift_calc_keypoint_descriptor
      vl_sift_calc_raw_descriptor=orig_vl_sift_calc_raw_descriptor
      vl_sift_keypoint_init=orig_vl_sift_keypoint_init
      vl_imconvcol_v=orig_vl_imconvcol_v
      vl_imconvcol_vf=orig_vl_imconvcol_vf
  )

  target_compile_options(vlfeat_original_reference
    PRIVATE
      $<$<COMPILE_LANG_AND_ID:C,GNU,Clang>:-Wno-unused-function>
  )

  set(COLITE_VLFEAT_REFERENCE_TARGET vlfeat_original_reference)
else()
  message(STATUS
    "VLFeat reference tree not found: reference-consistency tests are disabled")
endif()

add_executable(colite_core_tests ${COLITE_CORE_TEST_SOURCES})
if(MSVC)
  target_sources(colite_core_tests PRIVATE "${PROJECT_SOURCE_DIR}/cmake/windows.manifest")
endif()

if(COLITE_ENABLE_CUDA_SIFT)
  add_executable(colite_sift_resource_probe benchmarks/sift_resource_probe.cpp)
  target_link_libraries(colite_sift_resource_probe PRIVATE colite_command_exe)
endif()

target_link_libraries(colite_core_tests
  PRIVATE
    colite_core
    colite_command_exe
    colite_sfm_command_exe
    colite_recon_mvs_exe
    colite_recon_sfm_exe
    GTest::gtest_main
    ${COLITE_VLFEAT_REFERENCE_TARGET}
)

target_include_directories(colite_core_tests
  PRIVATE
    ${PROJECT_SOURCE_DIR}/thirdparty
    ${PROJECT_SOURCE_DIR}/src/colite/features/backends
    ${PROJECT_SOURCE_DIR}/src
)

if(COLITE_VLFEAT_REFERENCE_TARGET)
  target_include_directories(colite_core_tests
    PRIVATE
      ${COLITE_VLFEAT_REFERENCE_DIR}/vl
  )
endif()

target_compile_definitions(colite_core_tests
  PRIVATE
    COLITE_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)

include(GoogleTest)
gtest_discover_tests(colite_core_tests)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_test(NAME LoggingCliTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/scripts/logging_cli_test.py)
set_tests_properties(LoggingCliTest PROPERTIES
  ENVIRONMENT "COLITE_EXE=$<TARGET_FILE:colite>;PYTHONDONTWRITEBYTECODE=1")

add_test(NAME SfmBuildIdentityTest
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/scripts/sfm_build_identity_test.py)
set_tests_properties(SfmBuildIdentityTest PROPERTIES
  ENVIRONMENT "COLITE_EXE=$<TARGET_FILE:colite>;PYTHONDONTWRITEBYTECODE=1"
  SKIP_RETURN_CODE 77)

add_test(
  NAME ColmapDatabaseImportTest
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/tests/scripts/colmap_database_import_test.py
)
set_tests_properties(
  ColmapDatabaseImportTest
  PROPERTIES
    ENVIRONMENT "COLITE_EXE=$<TARGET_FILE:colite>;PYTHONDONTWRITEBYTECODE=1"
    SKIP_RETURN_CODE 77
)
