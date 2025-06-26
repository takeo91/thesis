# Sensor Type/Location Identification Research Questions
## Analysis & Implementation Feasibility

### **Primary Research Questions**

#### 1. **"What type of sensor generated this unknown window?"**
- **Goal**: Predict sensor modality (Accelerometer vs Gyroscope vs Magnetometer vs IMU)
- **Implementation**: Multi-class classification using sensor data features
- **Feasibility**: 🟢 **Easy** - Opportunity dataset has clear sensor type labels in column metadata
- **Data Available**: Column names indicate sensor types (acc, gyro, magnetic, quaternion)
- **Expected Accuracy**: High (90%+) - different sensor types have distinct signal characteristics

#### 2. **"Where on the body was this sensor placed?"**
- **Goal**: Predict body location (Ankle vs Wrist vs Chest vs Back vs Hip vs Upper Arm)
- **Implementation**: Body part classification using movement pattern analysis
- **Feasibility**: 🟢 **Easy** - Column naming clearly indicates body parts (RKN, LUA, RWR, BACK)
- **Data Available**: 120 sensor channels with body part prefixes
- **Expected Accuracy**: High (85%+) - different body locations have distinct movement signatures

#### 3. **"Which specific sensor location is this?"**
- **Goal**: Fine-grained location prediction (RKN vs LKN vs RUA vs LUA vs RWR vs LWR)
- **Implementation**: Multi-class classification with 15-20 specific sensor locations
- **Feasibility**: 🟡 **Medium** - More challenging due to similar locations (left vs right arm)
- **Data Available**: Full sensor location mapping available in column_names.txt
- **Expected Accuracy**: Medium (70-80%) - left/right disambiguation is challenging

### **Comparative Analysis Questions**

#### 4. **"How well can we distinguish ankle sensors from wrist sensors?"**
- **Goal**: Binary classification performance between specific body location pairs
- **Implementation**: Pairwise classification experiments across all body part combinations
- **Feasibility**: 🟢 **Easy** - Can reuse existing per-sensor similarity framework
- **Expected Results**: High accuracy (95%+) for distant locations, lower for similar locations

#### 5. **"Which sensor types are most distinguishable from each other?"**
- **Goal**: Confusion matrix analysis for sensor type discrimination
- **Implementation**: Multi-class sensor type classification with detailed error analysis
- **Feasibility**: 🟢 **Easy** - Direct application of existing classification pipeline
- **Expected Insights**: Accelerometer vs Gyroscope likely highly distinguishable

#### 6. **"Can we identify sensor placement without knowing the activity?"**
- **Goal**: Activity-agnostic sensor location identification
- **Implementation**: Train/test across mixed activity types
- **Feasibility**: 🟡 **Medium** - Requires careful cross-activity validation
- **Challenge**: Some sensors may be more discriminative during specific activities

### **Cross-Activity Robustness Questions**

#### 7. **"Does sensor identification work across different activities?"**
- **Goal**: Train on one activity type, test sensor ID on different activities
- **Implementation**: Cross-activity generalization experiments
- **Feasibility**: 🟡 **Medium** - Can leverage existing multi-label type framework
- **Data Available**: 3 activity hierarchies (Locomotion, ML_Both_Arms, HL_Activity)
- **Expected Challenge**: Ankle sensors during "walking" vs "sitting" may have different signatures

#### 8. **"Which body locations have the most distinctive sensor signatures?"**
- **Goal**: Rank body locations by identification ease/reliability
- **Implementation**: Per-location classification accuracy analysis
- **Feasibility**: 🟢 **Easy** - Direct analysis of existing per-sensor results
- **Expected Results**: Ankle/leg sensors likely most distinctive due to gait patterns

#### 9. **"How does activity type affect sensor identification accuracy?"**
- **Goal**: Activity-specific sensor identification performance
- **Implementation**: Stratified analysis by activity label
- **Feasibility**: 🟡 **Medium** - Requires combining activity and sensor classification
- **Insights**: Locomotion activities likely easier for leg sensors, manipulation for arm sensors

### **Advanced Questions (Extensions)**

#### 10. **"Given a mystery sensor stream, what's the most likely body placement?"**
- **Goal**: Ranking-based sensor location prediction (Top-3 most likely locations)
- **Implementation**: Multi-class classification with confidence scoring
- **Feasibility**: 🟢 **Easy** - Can reuse existing similarity ranking framework
- **Output**: Ranked list of sensor locations with confidence scores

#### 11. **"Can we detect if a sensor has been moved to a different body location?"**
- **Goal**: Sensor displacement detection/anomaly detection
- **Implementation**: Train on known placements, detect distribution shifts
- **Feasibility**: 🔴 **Hard** - Requires temporal modeling and anomaly detection
- **Challenge**: Need baseline "normal" behavior for each sensor location

#### 12. **"Which sensor characteristics are most informative for location identification?"**
- **Goal**: Feature importance analysis for sensor location prediction
- **Implementation**: Feature analysis using existing 38 similarity metrics
- **Feasibility**: 🟡 **Medium** - Can leverage existing metric framework
- **Insights**: Movement amplitude, frequency patterns, correlation structures

### **Multi-Modal Sensor Questions**

#### 13. **"Can we identify sensor location using only accelerometer data?"**
- **Goal**: Single-modality sensor location identification
- **Implementation**: Filter to accelerometer columns only, repeat classification
- **Feasibility**: 🟢 **Easy** - Simple data filtering + existing pipeline
- **Practical Value**: Cost-effective sensor deployment strategies

#### 14. **"Do we need full IMU data or is accelerometer sufficient for location ID?"**
- **Goal**: Cost-benefit analysis of sensor complexity
- **Implementation**: Performance comparison across sensor modality subsets
- **Feasibility**: 🟢 **Easy** - Multiple experiments with different sensor subsets
- **Business Impact**: Hardware cost optimization

#### 15. **"How many sensors do we need for reliable body location mapping?"**
- **Goal**: Minimum sensor requirements for location identification
- **Implementation**: Progressive sensor reduction experiments
- **Feasibility**: 🟡 **Medium** - Requires combinatorial sensor subset analysis
- **Challenge**: Exponential number of sensor combinations

### **Real-World Deployment Questions**

#### 16. **"How robust is sensor identification to different people?"**
- **Goal**: Person-independent sensor location identification
- **Implementation**: Cross-subject validation (Opportunity has 4 subjects)
- **Feasibility**: 🟡 **Medium** - Limited by 4 subjects in Opportunity dataset
- **Data Limitation**: Only 4 subjects may not be sufficient for robust analysis

#### 17. **"Can we identify sensor misplacement during deployment?"**
- **Goal**: Quality control for sensor network deployment
- **Implementation**: Outlier detection for sensor behavior patterns
- **Feasibility**: 🔴 **Hard** - Requires extensive normal behavior modeling
- **Practical Value**: High for real sensor network deployments

#### 18. **"Which sensor placements are most prone to identification errors?"**
- **Goal**: Error analysis for commonly confused sensor locations
- **Implementation**: Confusion matrix analysis + error pattern identification
- **Feasibility**: 🟢 **Easy** - Direct analysis of classification results
- **Expected Results**: Left/right arm confusion, similar body regions

### **Implementation Priority & Difficulty Assessment**

#### **🟢 Easy (Can implement immediately)**
1. Basic sensor type identification (Q1)
2. Body location classification (Q2)
3. Pairwise location discrimination (Q4)
4. Sensor type confusion analysis (Q5)
5. Single-modality experiments (Q13, Q14)
6. Error pattern analysis (Q18)

#### **🟡 Medium (Requires additional development)**
7. Fine-grained location prediction (Q3)
8. Cross-activity validation (Q7, Q9)
9. Feature importance analysis (Q12)
10. Cross-subject validation (Q16)
11. Minimum sensor requirements (Q15)

#### **🔴 Hard (Significant research extensions)**
12. Temporal anomaly detection (Q11)
13. Sensor misplacement detection (Q17)
14. Activity-agnostic robust identification (Q6)

### **Recommended Starting Point**
Begin with **Questions 1, 2, 4, 5, 13** as they can leverage your existing per-sensor framework with minimal modifications and will provide immediate insights into sensor identification feasibility.

---

## **Data Requirements & Availability**

### **Opportunity Dataset Sensor Metadata**
- **Sensor Types**: Accelerometer, Gyroscope, Magnetometer, IMU, REED switches
- **Body Locations**: 
  - **Upper Body**: RUA/LUA (upper arms), RLA/LLA (lower arms), RWR/LWR (wrists)
  - **Lower Body**: RKN/LKN (knees), R-SHOE/L-SHOE (feet)
  - **Torso**: HIP, BACK
- **Total Sensors**: 120 channels across multiple sensor types and locations
- **Subjects**: 4 participants with multiple activity sessions each

### **Implementation Strategy**

#### **Phase 1: Foundation (Easy Questions)**
1. **Sensor Type Classification**: Use column metadata to create sensor type labels
2. **Body Location Classification**: Extract body part from column names
3. **Baseline Performance**: Establish accuracy benchmarks for both tasks

#### **Phase 2: Comparative Analysis (Easy-Medium Questions)**
1. **Pairwise Discrimination**: Test all body location pairs
2. **Confusion Analysis**: Identify commonly confused sensors/locations
3. **Single-Modality Tests**: Compare accelerometer-only vs full IMU performance

#### **Phase 3: Advanced Analysis (Medium-Hard Questions)**
1. **Cross-Activity Validation**: Train on one activity type, test on others
2. **Feature Importance**: Identify which similarity metrics matter most
3. **Cross-Subject Validation**: Test generalization across different people

### **Expected Research Contributions**

1. **Methodological**: Novel application of fuzzy similarity metrics to sensor identification
2. **Practical**: Guidelines for optimal sensor placement in body sensor networks
3. **Technical**: Performance benchmarks for sensor type/location identification
4. **Economic**: Cost-benefit analysis of sensor complexity requirements

### **Integration with Existing Work**

This sensor identification research complements your activity recognition work by:
- **Validating Sensor Choices**: Confirming which sensors are most informative
- **Quality Control**: Detecting sensor placement errors in real deployments
- **Cost Optimization**: Identifying minimum sensor requirements
- **Robustness Analysis**: Understanding sensor identification reliability across conditions