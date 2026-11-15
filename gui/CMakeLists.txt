cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  include(../cmake/ColiteVersion.cmake)
  project(colite_workbench VERSION ${COLITE_VERSION} LANGUAGES CXX)
  option(COLITE_BUILD_TESTS "Build unit tests" ON)
endif()
if(COLITE_BUILD_TESTS)
  enable_testing()
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
find_package(Qt5 5.15.2 EXACT REQUIRED COMPONENTS Core Gui Widgets Concurrent)
find_package(OpenGL REQUIRED)
find_package(PNG REQUIRED)
add_library(colite_gui_core
  project/project.cpp
  project/recovery_store.cpp
  project/scene_export.cpp
  project/workspace.cpp
  workflow/artifact_validator.cpp
  workflow/parameter_schema.cpp
  workflow/task_plan.cpp
  workflow/task_runner.cpp
  workflow/workflow_state.cpp
)
target_link_libraries(colite_gui_core PUBLIC Qt5::Core Qt5::Gui Qt5::Concurrent PNG::PNG)
target_include_directories(colite_gui_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} "${CMAKE_BINARY_DIR}/generated")
set(colite_gui_ui_sources
  app/main_window.cpp
  widgets/feature_view.cpp
  widgets/inspector_panel.cpp
  widgets/parameter_dialog.cpp
  widgets/scene_view.cpp
  assets/ui.qrc
)
add_executable(colite_gui WIN32 app/main.cpp ${colite_gui_ui_sources})
if(WIN32)
  target_sources(colite_gui PRIVATE windows.rc "${CMAKE_CURRENT_SOURCE_DIR}/../cmake/windows.manifest")
endif()
target_link_libraries(colite_gui PRIVATE colite_gui_core Qt5::Widgets Qt5::Concurrent OpenGL::GL)
set(gui_targets colite_gui_core colite_gui)
if(COLITE_BUILD_TESTS)
  find_package(Qt5 5.15.2 EXACT REQUIRED COMPONENTS Test)
  add_executable(colite_gui_tests tests/gui_tests.cpp)
  target_link_libraries(colite_gui_tests PRIVATE colite_gui_core Qt5::Test)
  add_test(NAME colite_gui_contracts COMMAND colite_gui_tests -o -,txt
    -o "${CMAKE_CURRENT_BINARY_DIR}/gui-contracts.xml,junitxml")
  add_executable(colite_gui_ui_tests tests/ui_tests.cpp ${colite_gui_ui_sources})
  target_link_libraries(colite_gui_ui_tests PRIVATE colite_gui_core Qt5::Widgets Qt5::Test OpenGL::GL)
  add_test(NAME colite_gui_inspection COMMAND colite_gui_ui_tests -o -,txt
    -o "${CMAKE_CURRENT_BINARY_DIR}/gui-inspection.xml,junitxml")
  set_tests_properties(colite_gui_inspection PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen")
  list(APPEND gui_targets colite_gui_tests colite_gui_ui_tests)
endif()
foreach(target IN LISTS gui_targets)
  if(MSVC)
    target_compile_options(${target} PRIVATE /utf-8 /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()
install(TARGETS colite_gui RUNTIME DESTINATION bin)
install(FILES workflow/worker_bridge.py DESTINATION bin)
install(FILES assets/PHOSPHOR-LICENSE DESTINATION share/colite/licenses/Phosphor)
