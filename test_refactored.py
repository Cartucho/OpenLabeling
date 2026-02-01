"""
Simple test script to verify the refactored OpenLabeling code
"""
import os
import sys

# Add the refactored directory to the path
sys.path.insert(0, '/workspace/refactored_openlabeling')

def test_imports():
    """Test that all modules can be imported without errors"""
    try:
        from openlabeling.config import Config
        print("✓ Config module imported successfully")
        
        from openlabeling.utils import point_in_rect, yolo_format
        print("✓ Utils module imported successfully")
        
        from openlabeling.bbox_handler import DragBoundingBox, BoundingBoxHandler
        print("✓ Bbox handler module imported successfully")
        
        from openlabeling.tracker import LabelTracker
        print("✓ Tracker module imported successfully")
        
        from openlabeling.app import OpenLabelingApp
        print("✓ App module imported successfully")
        
        return True
    except Exception as e:
        print(f"✗ Error importing modules: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without running the full app"""
    try:
        # Test utility functions
        from openlabeling.utils import point_in_rect, yolo_format, get_bbox_area
        
        # Test point_in_rect
        result = point_in_rect(5, 5, 0, 0, 10, 10)
        assert result == True, "point_in_rect failed"
        print("✓ point_in_rect function works correctly")
        
        # Test get_bbox_area
        area = get_bbox_area(0, 0, 10, 10)
        assert area == 100, f"get_bbox_area failed, got {area}"
        print("✓ get_bbox_area function works correctly")
        
        # Test yolo_format
        yolo_str = yolo_format(0, (10, 10), (50, 50), 100, 100)
        print(f"✓ yolo_format: {yolo_str}")
        
        # Test configuration
        import tempfile
        import argparse
        from unittest.mock import patch
        
        # Mock command line arguments
        with patch('argparse.ArgumentParser.parse_args') as mock_parse:
            mock_parse.return_value = argparse.Namespace(
                input_dir='/tmp',
                output_dir='/tmp/output',
                thickness=2,
                draw_from_PASCAL_files=False,
                tracker='KCF',
                n_frames=100
            )
            
            from openlabeling.config import Config
            config = Config()
            print(f"✓ Config created: delay={config.delay}, with_qt={config.with_qt}")
        
        return True
    except Exception as e:
        print(f"✗ Error testing basic functionality: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("Testing refactored OpenLabeling modules...\n")
    
    print("1. Testing imports:")
    if not test_imports():
        print("\nImport tests failed. Stopping.")
        return
    
    print("\n2. Testing basic functionality:")
    if not test_basic_functionality():
        print("\nBasic functionality tests failed.")
        return
    
    print("\n✓ All tests passed! The refactored code appears to work correctly.")

if __name__ == "__main__":
    main()