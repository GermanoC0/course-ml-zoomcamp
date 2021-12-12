#!/usr/bin/env python
# coding: utf-8

import requests

# URL of the service
url = 'http://localhost:9090/classify'

# JOB POST data
job_post = {'location_available': 'available',
 'department_available': 'not_available',
 'salary_range_available': 'available',
 'company_profile_available': 'not_available',
 'description_available': 'available',
 'requirements_available': 'available',
 'benefits_available': 'available',
 'employment_type': 'full-time',
 'required_experience': 'entry_level',
 'required_education': 'high_school_or_equivalent',
 'industry': 'telecommunications',
 'function': 'customer_service',
 'telecommuting': 0,
 'has_company_logo': 0,
 'has_questions': 0,
 'company_profile_lenght': 0,
 'description_lenght': 564,
 'requirements_lenght': 202,
 'benefits_lenght': 153,
 'title_lenght': 26}


#job_post = {'location_available': 'available',
# 'department_available': 'available',
# 'salary_range_available': 'not_available',
# 'company_profile_available': 'available',
# 'description_available': 'available',
# 'requirements_available': 'available',
# 'benefits_available': 'available',
# 'employment_type': 'full-time',
# 'required_experience': 'mid-senior_level',
# 'required_education': "bachelor's_degree",
# 'industry': 'telecommunications',
# 'function': 'product_management',
# 'telecommuting': 0,
# 'has_company_logo': 1,
# 'has_questions': 0,
# 'company_profile_lenght': 2676,
# 'description_lenght': 5652,
# 'requirements_lenght': 2290,
# 'benefits_lenght': 225,
# 'title_lenght': 32}

# Response from the service
response = requests.post(url, json=job_post).json()
print(response)

if response['fraudulent'] >= 0.87:
    print('Better not consider this job offer ...')
else:
    print('This job offer seems to be ok ...')

