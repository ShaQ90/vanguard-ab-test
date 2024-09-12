# VANGUARD A/B TESTING PROJECT

#### Project Overview
This project aims to evaluate the effectiveness of a redesigned user interface (UI) for Vanguard’s online platform. Through A/B testing, we analyzed whether the new design led to a higher completion rate compared to the traditional interface. Using data from the Control and Test groups, we conducted detailed exploratory data analysis (EDA), statistical testing, and visualization to assess user behavior and performance metrics.
The main goal was to determine if the new UI increased user completion rates and provided a more efficient user experience.

#### Research Questions
The key objectives and questions that guided our analysis include:
Completion Rate: Did the new UI lead to a higher 1. completion rate compared to the control group?
3. Time Efficiency: Did users in the test group complete steps faster than those in the control group?
5. User Demographics: Did demographic factors such as age, gender, or number of accounts influence the completion rate in the test group?
7. Support Requests: Were there fewer support calls (errors) for users in the test group compared to the control group?

#### Datasets
We utilized several datasets provided by Vanguard to perform this analysis. The datasets include:
1. Client Profiles: This contains demographic information such as age, gender, number of accounts, and account tenure.
3. Digital Footprints: Web interaction data that tracks each user’s step-by-step journey on the platform, including process steps and timestamps.
5. Experiment Roster: Information on which clients were assigned to the control or test group for the A/B test.

##### Data Challenges and Solutions
Throughout the analysis, we encountered some data quality issues:
- Missing Values: Some data points, particularly demographic information, had missing values. These were handled using median imputation for numerical fields and mode imputation for categorical fields.
- Duplicate Entries: Duplicate rows were identified and removed to ensure data accuracy.
- Data Merging: The Client Profiles and Experiment Roster datasets were merged using client_id to consolidate information for analysis.

#### Exploratory Data Analysis (EDA)
To better understand the behavior of the users in both the test and control groups, we conducted the following EDA steps:
Completion Rates: We visualized and compared the completion rates for both groups to evaluate if the new design improved user engagement.
Time Spent on Steps: We analyzed the average time users spent at each process step to assess if the new UI reduced completion time.
Demographic Analysis: We explored the influence of demographics, such as age, gender, and the number of accounts, on the performance of the test group compared to the control group.

#### Statistical Testing
To draw data-driven conclusions, we conducted hypothesis tests to evaluate the success of the new design.

###### Key Hypothesis:
- Null Hypothesis (H₀): The completion rates of the control and test groups are equal.
- Alternative Hypothesis (H₁): The completion rate of the test group is higher than that of the control group by at least 5%.
We performed a proportion test to compare completion rates, and the resulting p-value (< 0.05) allowed us to reject the null hypothesis, showing that the test group’s completion rate was significantly higher.

#### Key Findings and Insights
1. Completion Rate: The test group achieved a significantly higher completion rate compared to the control group, suggesting that the new UI is more effective.
3. Time Efficiency: Users in the test group completed the process steps faster, especially in the later steps, further supporting the hypothesis that the redesign improves user experience.
5. Support Requests: The number of support calls (errors) was lower in the test group, indicating that the new design led to fewer user issues.
7. Demographic Influence: The redesign seems to have benefited all demographic groups, with slight variations in how users with multiple accounts or different genders interacted with the new interface.

#### Conclusion and Recommendations
Based on the findings, we recommend adopting the redesigned UI as it leads to a higher completion rate, faster process times, and fewer user issues. Future analyses could explore whether additional tweaks to the design could further improve user satisfaction or if the design performs differently for specific demographic subgroups.

##### Next Steps:
1. A/B Testing in Additional Markets: Conduct similar tests in other regions or product categories.
3. Deeper Demographic Analysis: Explore the impact of the redesign on users based on specific demographics such as account tenure and activity level.
5. Ongoing Optimization: Continuously monitor the new UI’s performance and make iterative improvements.

### Repository Structure
bash
Code kopieren
/data
    /clean
    user_data.csv
    user_logs.csv
    /raw
    df_final_demo.txt
    df_final_experiment_clients.txt
    df_final_web_data_pt_1.txt
    df_final_web_data_pt_2.txt
/notebooks
   data_clean_testing.ipynb
   functions.py
   README.md
This repository contains the raw and clean data files, the analysis notebooks, and this README explaining the project. The notebooks are organized by key analysis steps such as EDA, and hypothesis testing.

####
 
Kanban: https://trello.com/invite/b/66d5c9165eaace0e0c397fe0/ATTI4954520f88112d22cc4fe8e20e60c612E66B0065/project-2

Presentation: https://docs.google.com/presentation/d/1qe0GqeAvFDE_mzD351KraOdM8xz-KGxjVv6bZGcsgQc/edit?usp=sharing
