"""
Enhanced Data Validator and Preprocessor for Temple Data
Provides comprehensive data validation, cleaning, and enrichment
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TempleDataValidator:
    """Comprehensive data validator and preprocessor for temple information"""
    
    def __init__(self):
        self.required_columns = [
            'templeName', 'Description', 'Location', 'State', 
            'Coordinates', 'Latitude', 'Longitude'
        ]
        
        self.coordinate_bounds = {
            'lat_min': 6.0,   # Southernmost point of India
            'lat_max': 37.0,  # Northernmost point of India
            'lon_min': 68.0,  # Westernmost point of India
            'lon_max': 98.0   # Easternmost point of India
        }
        
        self.validation_errors = []
        self.warnings = []
        
    def validate_dataframe(self, df: pd.DataFrame) -> Dict:
        """Comprehensive validation of temple DataFrame"""
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'stats': {},
            'cleaned_data': None
        }
        
        try:
            # Check basic structure
            structure_check = self._validate_structure(df)
            if not structure_check['is_valid']:
                validation_results['is_valid'] = False
                validation_results['errors'].extend(structure_check['errors'])
                return validation_results
            
            # Create copy for cleaning
            df_clean = df.copy()
            
            # Validate and clean coordinates
            coord_results = self._validate_coordinates(df_clean)
            validation_results['warnings'].extend(coord_results['warnings'])
            validation_results['errors'].extend(coord_results['errors'])
            
            # Validate temple names
            name_results = self._validate_temple_names(df_clean)
            validation_results['warnings'].extend(name_results['warnings'])
            
            # Validate descriptions
            desc_results = self._validate_descriptions(df_clean)
            validation_results['warnings'].extend(desc_results['warnings'])
            
            # Validate locations
            location_results = self._validate_locations(df_clean)
            validation_results['warnings'].extend(location_results['warnings'])
            
            # Clean and standardize data
            df_clean = self._clean_and_standardize(df_clean)
            
            # Generate statistics
            validation_results['stats'] = self._generate_statistics(df_clean)
            validation_results['cleaned_data'] = df_clean
            
            # Final validation check
            if len(validation_results['errors']) > 0:
                validation_results['is_valid'] = False
            
            logger.info(f"Validation completed. Valid: {validation_results['is_valid']}, "
                       f"Errors: {len(validation_results['errors'])}, "
                       f"Warnings: {len(validation_results['warnings'])}")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            validation_results['is_valid'] = False
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            return validation_results
    
    def _validate_structure(self, df: pd.DataFrame) -> Dict:
        """Validate basic DataFrame structure"""
        errors = []
        
        if df.empty:
            errors.append("DataFrame is empty")
            return {'is_valid': False, 'errors': errors}
        
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        return {'is_valid': len(errors) == 0, 'errors': errors}
    
    def _validate_coordinates(self, df: pd.DataFrame) -> Dict:
        """Validate geographical coordinates"""
        errors = []
        warnings = []
        
        for idx, row in df.iterrows():
            try:
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                
                # Check if coordinates are within India bounds
                if not (self.coordinate_bounds['lat_min'] <= lat <= self.coordinate_bounds['lat_max']):
                    warnings.append(f"Row {idx}: Latitude {lat} may be outside India bounds")
                
                if not (self.coordinate_bounds['lon_min'] <= lon <= self.coordinate_bounds['lon_max']):
                    warnings.append(f"Row {idx}: Longitude {lon} may be outside India bounds")
                
                # Update coordinate string format
                df.loc[idx, 'Coordinates'] = f"{lat:.4f}, {lon:.4f}"
                
            except (ValueError, TypeError) as e:
                errors.append(f"Row {idx}: Invalid coordinates - {e}")
        
        return {'errors': errors, 'warnings': warnings}
    
    def _validate_temple_names(self, df: pd.DataFrame) -> Dict:
        """Validate and standardize temple names"""
        warnings = []
        
        for idx, row in df.iterrows():
            name = str(row['templeName']).strip()
            
            # Check for duplicates
            duplicates = df[df['templeName'].str.strip() == name]
            if len(duplicates) > 1:
                warnings.append(f"Possible duplicate temple name: {name}")
            
            # Standardize name format
            df.loc[idx, 'templeName'] = self._standardize_temple_name(name)
        
        return {'warnings': warnings}
    
    def _validate_descriptions(self, df: pd.DataFrame) -> Dict:
        """Validate temple descriptions"""
        warnings = []
        
        for idx, row in df.iterrows():
            desc = str(row['Description']).strip()
            
            if len(desc) < 50:
                warnings.append(f"Row {idx}: Description may be too short ({len(desc)} chars)")
            elif len(desc) > 2000:
                warnings.append(f"Row {idx}: Description may be too long ({len(desc)} chars)")
            
            # Clean description
            df.loc[idx, 'Description'] = self._clean_description(desc)
        
        return {'warnings': warnings}
    
    def _validate_locations(self, df: pd.DataFrame) -> Dict:
        """Validate location information"""
        warnings = []
        
        # Check for consistent state naming
        state_counts = df['State'].value_counts()
        for state, count in state_counts.items():
            if count == 1:
                warnings.append(f"Only one temple found in {state}, verify location accuracy")
        
        return {'warnings': warnings}
    
    def _standardize_temple_name(self, name: str) -> str:
        """Standardize temple name format"""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Ensure proper capitalization
        name = name.title()
        
        return name
    
    def _clean_description(self, description: str) -> str:
        """Clean and standardize description text"""
        # Remove extra whitespace
        description = re.sub(r'\s+', ' ', description.strip())
        
        # Fix common issues
        description = description.replace(' ,', ',')
        description = description.replace(' .', '.')
        description = re.sub(r'\.+', '.', description)
        
        # Ensure it ends with a period
        if not description.endswith('.'):
            description += '.'
        
        return description
    
    def _clean_and_standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply final cleaning and standardization"""
        df_clean = df.copy()
        
        # Standardize numerical columns
        numerical_cols = ['DistanceFromMumbai_Km', 'DistanceFromNewDelhi_Km', 
                         'DistanceFromChennai_Km', 'DistanceFromKolkata_Km']
        
        for col in numerical_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                df_clean[col] = df_clean[col].round(0).astype('Int64')
        
        # Standardize categorical columns
        categorical_cols = ['State', 'Category', 'Architecture']
        for col in categorical_cols:
            if col in df_clean.columns:
                df_clean[col] = df_clean[col].str.strip().str.title()
        
        return df_clean
    
    def _generate_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive statistics about the dataset"""
        stats = {
            'total_temples': len(df),
            'states_covered': df['State'].nunique() if 'State' in df.columns else 0,
            'avg_description_length': df['Description'].str.len().mean() if 'Description' in df.columns else 0,
            'coordinate_coverage': (~df[['Latitude', 'Longitude']].isna()).all(axis=1).sum() if all(col in df.columns for col in ['Latitude', 'Longitude']) else 0,
            'missing_data': {}
        }
        
        # Calculate missing data for each column
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                stats['missing_data'][col] = {
                    'count': int(missing_count),
                    'percentage': round((missing_count / len(df)) * 100, 2)
                }
        
        return stats
    
    def export_validation_report(self, validation_results: Dict, output_path: str = None) -> str:
        """Export validation report to JSON"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'validation_summary': {
                'is_valid': validation_results['is_valid'],
                'total_errors': len(validation_results['errors']),
                'total_warnings': len(validation_results['warnings'])
            },
            'errors': validation_results['errors'],
            'warnings': validation_results['warnings'],
            'statistics': validation_results['stats']
        }
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return f"Report saved to {output_path}"
        else:
            return json.dumps(report, indent=2, ensure_ascii=False)

def validate_temples_csv(file_path: str) -> Dict:
    """Main function to validate temple CSV file"""
    try:
        # Load data
        df = pd.read_csv(file_path)
        
        # Initialize validator
        validator = TempleDataValidator()
        
        # Perform validation
        results = validator.validate_dataframe(df)
        
        return results
        
    except Exception as e:
        logger.error(f"Error loading or validating file {file_path}: {e}")
        return {
            'is_valid': False,
            'errors': [f"Failed to load file: {str(e)}"],
            'warnings': [],
            'stats': {},
            'cleaned_data': None
        }

if __name__ == "__main__":
    # Example usage
    results = validate_temples_csv("temples_optimized.csv")
    
    print(f"Validation Status: {'PASSED' if results['is_valid'] else 'FAILED'}")
    print(f"Errors: {len(results['errors'])}")
    print(f"Warnings: {len(results['warnings'])}")
    
    if results['cleaned_data'] is not None:
        print(f"Total temples: {len(results['cleaned_data'])}")
        
        # Save cleaned data
        results['cleaned_data'].to_csv("temples_validated.csv", index=False)
        print("Cleaned data saved to temples_validated.csv") 