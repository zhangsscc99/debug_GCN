# adj matrix creation
def adj_matrix(AU_inc1,AU_inc2,feature):
    cnt1 = 0
    cnt2 = 0
    cnt_joint = 0

    for i in range(len(feature)):
        if  feature[i][AU_inc1] == 1.0:
            cnt1 += 1
        if  feature[i][AU_inc2] == 1.0:
            cnt2 += 1
        if  feature[i][AU_inc2] == 1.0 and feature[i][AU_inc1]==1.0:
            cnt_joint += 1
    
    AU1_AU2_joint_count = cnt_joint # Number of instances where both AU1 and AU4 are present
    AU2_count = cnt2 # Number of instances where AU4 is present
    total_count = len(feature) # Total number of instances in the dataset

    P_AU1_AU2 = AU1_AU2_joint_count / total_count
    P_AU2 = AU2_count / total_count
    #P_AU1_given_AU2 = P_AU1_AU2 / P_AU2
    if AU2_count == 0.0:
        return 0.0
    P12 = AU1_AU2_joint_count/AU2_count
    return P12