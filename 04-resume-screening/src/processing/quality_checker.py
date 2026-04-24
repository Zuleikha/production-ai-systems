"""Data quality checking and validation utilities."""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
from src.models.schemas import ResumeSchema, DataQualityMetrics
from pydantic import ValidationError
import json
import hashlib
from pathlib import Path

class DataQualityChecker:
    """Comprehensive data quality checker for resume data."""
    
    def __init__(self):
        self.validation_errors = []
        self.quality_thresholds = {
            'completeness_min': 80.0,
            'duplicate_max': 5.0,
            'validity_min': 90.0
        }
    
    def validate_schema_compliance(self, data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Validate data against Pydantic schema with detailed error tracking."""
        valid_records = []
        invalid_records = []
        
        for idx, record in enumerate(data):
            try:
                # Validate against schema
                validated = ResumeSchema(**record)
                valid_records.append(validated.model_dump())
            except ValidationError as e:
                error_details = {
                    'record_index': idx,
                    'record_id': record.get('id', f'record_{idx}'),
                    'validation_errors': [
                        {
                            'field': '.'.join(str(loc) for loc in error['loc']),
                            'message': error['msg'],
                            'type': error['type'],
                            'input_value': str(error.get('input', 'N/A'))[:100]
                        }
                        for error in e.errors()
                    ]
                }
                invalid_records.append(error_details)
                self.validation_errors.extend(error_details['validation_errors'])
        
        return valid_records, invalid_records
    
    def calculate_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate field completeness percentages."""
        completeness = {}
        
        for column in df.columns:
            if df[column].dtype == 'object':
                # For string columns, check for non-null and non-empty
                complete_mask = df[column].notna() & (df[column].astype(str).str.strip() != '')
            else:
                # For numeric columns, just check for non-null
                complete_mask = df[column].notna()
            
            completeness[column] = (complete_mask.sum() / len(df)) * 100
        
        return completeness
    
    def detect_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect and analyze duplicate records."""
        if df.empty:
            return {'duplicate_count': 0, 'duplicate_percentage': 0.0}
        
        # Primary duplicate detection by name and email
        key_columns = ['name']
        if 'contact.email' in df.columns:
            key_columns.append('contact.email')
        elif any(col.endswith('.email') for col in df.columns):
            email_col = next(col for col in df.columns if col.endswith('.email'))
            key_columns.append(email_col)
        
        duplicates = df.duplicated(subset=key_columns, keep=False)
        duplicate_groups = df[duplicates].groupby(key_columns).size() if duplicates.any() else pd.Series()
        
        # Content-based duplicate detection using hashing
        content_hashes = df.apply(lambda row: hashlib.md5(
            str(sorted(row.dropna().to_dict().items())).encode()
        ).hexdigest(), axis=1)
        content_duplicates = content_hashes.duplicated()
        
        return {
            'duplicate_count': duplicates.sum(),
            'duplicate_percentage': (duplicates.sum() / len(df)) * 100,
            'duplicate_groups': duplicate_groups.to_dict(),
            'content_duplicate_count': content_duplicates.sum(),
            'unique_records': len(df) - duplicates.sum()
        }
    
    def analyze_data_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data distribution and patterns."""
        if df.empty:
            return {}
        
        analysis = {
            'record_count': len(df),
            'field_count': len(df.columns),
            'numeric_fields': df.select_dtypes(include=[np.number]).columns.tolist(),
            'text_fields': df.select_dtypes(include=['object']).columns.tolist(),
        }
        
        # Analyze key fields if they exist
        if 'skills' in df.columns:
            skills_lengths = df['skills'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            analysis['skills_stats'] = {
                'avg_skills_per_resume': skills_lengths.mean(),
                'min_skills': skills_lengths.min(),
                'max_skills': skills_lengths.max()
            }
        
        if 'experience' in df.columns:
            exp_lengths = df['experience'].apply(lambda x: len(x) if isinstance(x, list) else 0)
            analysis['experience_stats'] = {
                'avg_experience_count': exp_lengths.mean(),
                'min_experience': exp_lengths.min(),
                'max_experience': exp_lengths.max()
            }
        
        return analysis
    
    def check_data_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for data consistency issues."""
        consistency_issues = []
        
        # Check email format consistency
        if 'contact.email' in df.columns:
            invalid_emails = ~df['contact.email'].str.match(r'^[^@]+@[^@]+\.[^@]+$', na=False)
            if invalid_emails.any():
                consistency_issues.append({
                    'issue': 'invalid_email_format',
                    'count': invalid_emails.sum(),
                    'percentage': (invalid_emails.sum() / len(df)) * 100
                })
        
        # Check date format consistency
        date_columns = [col for col in df.columns if 'date' in col.lower()]
        for col in date_columns:
            if col in df.columns:
                invalid_dates = ~df[col].str.match(r'^\d{4}-\d{2}-\d{2}$|^present$', na=False)
                if invalid_dates.any():
                    consistency_issues.append({
                        'issue': f'invalid_date_format_{col}',
                        'count': invalid_dates.sum(),
                        'percentage': (invalid_dates.sum() / len(df)) * 100
                    })
        
        return {
            'consistency_issues': consistency_issues,
            'total_issues': len(consistency_issues)
        }
    
    def generate_comprehensive_report(self, 
                                    raw_data: List[Dict], 
                                    valid_data: List[Dict], 
                                    invalid_data: List[Dict]) -> DataQualityMetrics:
        """Generate comprehensive data quality report."""
        
        # Convert to DataFrame for analysis
        df = pd.json_normalize(valid_data) if valid_data else pd.DataFrame()
        
        # Calculate all metrics
        completeness = self.calculate_completeness(df) if not df.empty else {}
        duplicates_info = self.detect_duplicates(df) if not df.empty else {'duplicate_count': 0, 'duplicate_percentage': 0.0}
        distribution_info = self.analyze_data_distribution(df)
        consistency_info = self.check_data_consistency(df)
        
        # Create quality metrics object
        quality_metrics = DataQualityMetrics(
            total_records=len(raw_data),
            valid_records=len(valid_data),
            invalid_records=len(invalid_data),
            completeness_scores=completeness,
            duplicate_count=duplicates_info['duplicate_count'],
            duplicate_percentage=duplicates_info['duplicate_percentage'],
            validation_errors=invalid_data,
            quality_score=0.0  # Will be calculated by validator
        )
        
        # Add additional analysis to the report
        report_dict = quality_metrics.model_dump()
        report_dict.update({
            'data_distribution': distribution_info,
            'consistency_analysis': consistency_info,
            'quality_assessment': self._assess_quality(quality_metrics),
            'recommendations': self._generate_recommendations(quality_metrics, consistency_info)
        })
        
        return report_dict
    
    def _assess_quality(self, metrics: DataQualityMetrics) -> Dict[str, Any]:
        """Assess overall data quality against thresholds."""
        assessments = {}
        
        # Validity assessment
        validity_rate = (metrics.valid_records / metrics.total_records) * 100 if metrics.total_records > 0 else 0
        assessments['validity'] = {
            'score': validity_rate,
            'status': 'PASS' if validity_rate >= self.quality_thresholds['validity_min'] else 'FAIL',
            'threshold': self.quality_thresholds['validity_min']
        }
        
        # Completeness assessment
        avg_completeness = np.mean(list(metrics.completeness_scores.values())) if metrics.completeness_scores else 0
        assessments['completeness'] = {
            'score': avg_completeness,
            'status': 'PASS' if avg_completeness >= self.quality_thresholds['completeness_min'] else 'FAIL',
            'threshold': self.quality_thresholds['completeness_min']
        }
        
        # Duplicate assessment
        assessments['uniqueness'] = {
            'score': 100 - metrics.duplicate_percentage,
            'status': 'PASS' if metrics.duplicate_percentage <= self.quality_thresholds['duplicate_max'] else 'FAIL',
            'threshold': self.quality_thresholds['duplicate_max']
        }
        
        # Overall assessment
        passed_checks = sum(1 for assessment in assessments.values() if assessment['status'] == 'PASS')
        overall_score = (passed_checks / len(assessments)) * 100
        
        assessments['overall'] = {
            'score': overall_score,
            'status': 'EXCELLENT' if overall_score >= 90 else 'GOOD' if overall_score >= 70 else 'NEEDS_IMPROVEMENT',
            'checks_passed': f"{passed_checks}/{len(assessments)}"
        }
        
        return assessments
    
    def _generate_recommendations(self, metrics: DataQualityMetrics, consistency_info: Dict) -> List[str]:
        """Generate actionable recommendations for data quality improvement."""
        recommendations = []
        
        if metrics.invalid_records > 0:
            recommendations.append(f"Fix {metrics.invalid_records} records with validation errors")
        
        if metrics.duplicate_percentage > self.quality_thresholds['duplicate_max']:
            recommendations.append(f"Remove or merge {metrics.duplicate_count} duplicate records")
        
        if metrics.completeness_scores:
            incomplete_fields = [
                field for field, score in metrics.completeness_scores.items() 
                if score < self.quality_thresholds['completeness_min']
            ]
            if incomplete_fields:
                recommendations.append(f"Improve data completeness for fields: {', '.join(incomplete_fields)}")
        
        if consistency_info.get('total_issues', 0) > 0:
            recommendations.append("Address data consistency issues identified in the report")
        
        if not recommendations:
            recommendations.append("Data quality is excellent! No immediate actions required.")
        
        return recommendations