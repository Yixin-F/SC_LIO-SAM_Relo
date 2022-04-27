#include "utility.h"
#include "lio_sam/cloud_info.h"
#include "Scancontext.h"

#include<ros/ros.h>
#include<pcl/point_cloud.h>
#include<pcl_conversions/pcl_conversions.h>
#include<sensor_msgs/PointCloud2.h>
#include<pcl/io/pcd_io.h>
#include <pcl/registration/icp.h>

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <boost/algorithm/string.hpp>
#include <fstream>

/**  定义位姿类型 注册PointTypePose**/
struct PointXYZIRPYT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;                  // preferred way of adding a XYZ+padding
    float roll;
    float pitch;
    float yaw;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW   // make sure our new allocators are aligned
} EIGEN_ALIGN16;                    // enforce SSE padding for correct memory alignment

POINT_CLOUD_REGISTER_POINT_STRUCT (PointXYZIRPYT,
                                   (float, x, x) (float, y, y)
                                   (float, z, z) (float, intensity, intensity)
                                   (float, roll, roll) (float, pitch, pitch) (float, yaw, yaw)
                                   (double, time, time))

typedef PointXYZIRPYT  PointTypePose;

#define PI 3.1415926

class Relocation : public ParamServer
{
private:

    std::mutex read_mtx; 
    std::mutex mtx;
    

    SCManager scManagerRL;

    ifstream SCD_File;
    ifstream Pose_File;

    pcl::VoxelGrid<PointType> DownSample_sub;
    pcl::VoxelGrid<PointType> DownSample_relo;

    /***  先验地图  ***/
    ros::Publisher publishGlobalMap; 
    pcl::PointCloud<PointType>::Ptr globalMapCloud; 
    sensor_msgs::PointCloud2 GlobalMapMsg;

    /***  先验轨迹  ***/ 
    ros::Publisher publishGlobalTraj; // 轨迹
    pcl::PointCloud<PointType>::Ptr globalTrajCloud; 
    sensor_msgs::PointCloud2 GlobalTrajMsg;

    /***  待定位点云  ***/
    ros::Publisher publishReloCloud; // cloud
    pcl::PointCloud<PointType>::Ptr ReloCloud_result; 
    sensor_msgs::PointCloud2 ReloCloudMsg;

    ros::Publisher publishReloPose; // 轨迹
    PointTypePose ReloPose;
    nav_msgs::Odometry ReloPoseMsg;


    int preKey = -1; // 索引
    std::vector<cv::String> precloudName;
    PointTypePose transformPCD; // 将待定位点云水平变换一次
    // ros::Publisher publish2Relocation;
    // pcl::PointCloud<PointType>::Ptr cloud2Relocation;
    // sensor_msgs::PointCloud2 cloud2RelocationMsg;

    // image_transport::Publisher publishSCMat;
    // image_transport::ImageTransport it(nh); // 在"utility.h"
    cv::Mat SCImage;
    vector<Eigen::MatrixXd> curSC; // 当前扫描帧的SC
    vector<Eigen::MatrixXd> curRingkey;
    vector<Eigen::MatrixXd> curSectorkey;
    KeyMat cur_polarcontext_invkeys_mat;

    std::vector<int> curLoopKeyPre;
    std::vector<float> curYawDiffRad;

    pcl::PointCloud<PointType>::Ptr thiscloud_final;

    std::vector<int> relo_qureyIndex; // 匹配到的索引
    pcl::KdTreeFLANN<PointType>::Ptr searchNearestPose; // 找最近位姿，全局归一化路径

    /***  数据库点云  ***/
    std::string cloudPath = savePCDDirectory + "Scans/*.pcd"; 
    std::vector<cv::String> pcdFiles;
    std::string SCDPath = savePCDDirectory + "SCDs/*.scd";
    std::vector<cv::String> scdFiles;
    std::vector<std::pair<Eigen::Vector2f, Eigen::Vector2f>> dataBaseVector;

    /***  Pathplanner ***/
    pcl::PointCloud<PointType>::Ptr pathCloud;
    pcl::PointCloud<PointType> pathcloud_noptr;
    std::deque<int> pathsearch;
    std::deque<int> pathsearch_tmp;

    ros::Publisher publishPath; // 轨迹
    sensor_msgs::PointCloud2 PathMsg;

    std::vector<bool > if_new_path; // symbol of new path
    std::vector<std::pair<int, std::vector<int>>> all_path_part; // all parts of path

    
    /***  匹配到的点云及submap  ***/
    int queryNum = 3;
    std::vector<std::pair<std::string, std::vector<std::pair<int, double>>>> queryClouds;
    std::pair<std::string, std::vector<std::pair<int, double>>> tmpClouds; // "pre/query/yawdiff" to fill "queryCloud"
    
    int searchNum = 2;
    ros::Publisher publishsubMap;
    pcl::PointCloud<PointType>::Ptr subKeyframes;
    sensor_msgs::PointCloud2 subKeyframesMsg; 
    pcl::PointCloud<PointTypePose>::Ptr KeyFramePose2ICP;

public:
    Relocation(){

        publishGlobalMap = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/relocation/globalMapPCD", 5);
        publishGlobalTraj = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/relocation/globalTrajPCD", 5);

        publishsubMap = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/relocation/ICPsubmap", 5);
        publishReloCloud = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/relocation/ReloCloud", 5);
        publishReloPose = nh.advertise<nav_msgs::Odometry>("lio_sam/relocation/ReloPose", 5);

        publishPath = nh.advertise<sensor_msgs::PointCloud2>("lio_sam/pathplanner/PathGlobal", 5);
        
        // publishSCMat = it.advertise("lio_sam/relocation/SCImage", 1); // haven't succeeded
        
        allocateMemory();
        read(); // 读取地图或轨迹先验
    }
    
    std::string mode; // mode choice "relocation -> 1" or "pathplan -> 2" (argv[3])

    void allocateMemory(){

        /***  先验地图与轨迹  ***/
        globalMapCloud.reset(new pcl::PointCloud<PointType>());
        globalTrajCloud.reset(new pcl::PointCloud<PointType>());
        
        /***  待定位点云  ***/
        // cloud2Relocation.reset(new pcl::PointCloud<PointType>());
        ReloCloud_result.reset(new pcl::PointCloud<PointType>());
        thiscloud_final.reset(new pcl::PointCloud<PointType>());
        searchNearestPose.reset(new pcl::KdTreeFLANN<PointType>());
        transformPCD.x = transformPCD.y = transformPCD.z = 0; // 只含yaw旋转
        transformPCD.roll = transformPCD.pitch = transformPCD.yaw = 0;
        // transformPCD.x = transformPCD.y = 5;
        int k = 33.33;
        transformPCD.yaw = k * scManagerRL.PC_UNIT_SECTORANGLE / 180 * PI; // yaw尽量支持bin大小的倍数
        // transformPCD.yaw = 66 / 180 * PI;

        /***  数据库点云  ***/
        cv::glob(cloudPath, pcdFiles, false);
        cv::glob(SCDPath, scdFiles, false);

        relo_qureyIndex.resize(scdFiles.size());

        /***  匹配到的点云及submap  ***/
        // queryCloud.resize(queryNum);
        // queryCloudID.resize(queryNum);
        subKeyframes.reset(new pcl::PointCloud<PointType>());
        KeyFramePose2ICP.reset(new pcl::PointCloud<PointTypePose>());

        /***  pathplanner  ***/
        pathCloud.reset(new pcl::PointCloud<PointType>());
        pathsearch_tmp.resize(scdFiles.size());
        for(int i = 0; i < scdFiles.size(); i++){
            pathsearch_tmp[i] = i;
        }

        float filter_size = 0.5;  // to show
        DownSample_sub.setLeafSize(filter_size, filter_size, filter_size);
        DownSample_relo.setLeafSize(filter_size, filter_size, filter_size);
    }

    // void resetParams(){
    //     curSC.clear();
    //     curRingkey.clear();
    //     curSectorkey.clear();
    //     cur_polarcontext_invkeys_mat.clear();

    //     // queryCloud.clear();
    //     // for(int i = 0; i < queryCloud.size(); i++){
    //     //     queryCloud[i].reset(new pcl::PointCloud<PointType>());
    //     //     queryCloudID[i] = -1;
    //     // }
        
    // }

    /*
        功能：文件读取地图和位姿先验(PCD格式)
    */
    void read(){
        ROS_INFO("It maybe take a long time to load the map and trajectory");

        std::lock_guard<std::mutex> lock(read_mtx); // 自动锁
        
        if (pcl::io::loadPCDFile(savePCDDirectory + "cloudCorner.pcd", *globalMapCloud) == -1){
            ROS_ERROR("Can't load map");
            return ;
        }
        else ROS_INFO("Load map successfully");
        
        if (pcl::io::loadPCDFile(savePCDDirectory + "trajectory.pcd", *globalTrajCloud) == -1){
            ROS_ERROR("Can't load trajectory");
            return ;
        }
        else ROS_INFO("Load trajectory successfully");

        if (pcl::io::loadPCDFile(savePCDDirectory + "transformations.pcd", *KeyFramePose2ICP) == -1){
            ROS_ERROR("Can't load KeyFramePose6D");
            return ;
        }
        else{
            ROS_INFO("Load KeyFramePose6D successfully");
            // for(int i = 0; i < KeyFramePose2ICP->points.size(); i++){ // test z
            //     std::cout << KeyFramePose2ICP->points[i].z << std::endl; 
            // }
        }
    }

    /*
        功能：发布点云消息(点云地图或位姿)
    */
    void publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, sensor_msgs::PointCloud2 thisMsg, std::string thisFrame){

        if (thisCloud->points.size() == 0){
            ROS_ERROR("Can't load this file");
            return ;
        }
        pcl::toROSMsg(*thisCloud, thisMsg);
        thisMsg.header.stamp = ros::Time().now();
        thisMsg.header.frame_id = thisFrame;
        thisPub->publish(thisMsg);
    }

    /*
        功能：地图发布
    */
    void publishWorld(){
        ros::Rate rate(0.5);
        while(ros::ok()){
            publishCloud(&publishGlobalMap, globalMapCloud, GlobalMapMsg, "map"); // can open
            // publishCloud(&publishGlobalTraj, globalTrajCloud, GlobalTrajMsg, "map"); // can open
            // publishCloud(&publishsubMap, subKeyframes, subKeyframesMsg, "map"); // can open
            rate.sleep();
        }    
    }

    void PathPlaner(){
        pcl::PointCloud<PointType>::Ptr traj_copy(new pcl::PointCloud<PointType>()); 
        pcl::copyPointCloud(*globalTrajCloud, *traj_copy);

        pathcloud_noptr.width = traj_copy->points.size();
        pathcloud_noptr.height = 1;
        pathcloud_noptr.is_dense = false;
        pathcloud_noptr.resize(pathcloud_noptr.width * pathcloud_noptr.height);
        
        // surroundingkeyframeAddingDistThreshold = 1.0 关键帧判断距离阈值
        for(int i = 0; i < traj_copy->points.size(); i++){
            traj_copy->points[i].z = 0;  // z方向偏移为0
        }
        
        std::vector<int> hiahia_if(traj_copy->points.size(), 1); // 该关键帧位置是否进行归一化路径
        
        std::vector<int> searchInd;
        std::vector<float> searchSqDis;
        std::vector<int> toonearto_be1pose; // normalize these pose
        int pathInd = 0;

        searchNearestPose->setInputCloud(traj_copy); // kdtree

        for(int ind = 0; ind < traj_copy->points.size(); ind++){
            
            searchInd.clear(); // clear
            searchSqDis.clear();
            toonearto_be1pose.clear();

            if(hiahia_if[ind] == 1){

                pathsearch_tmp[ind] = ind;
                pathsearch.push_back(ind); // 查找索引更新

                searchNearestPose->radiusSearch(traj_copy->points[ind], (double)surroundingKeyframeSearchRadius * 5, searchInd, searchSqDis);

                for(int i = 0; i < searchInd.size(); i++){
                    if((abs(traj_copy->points[searchInd[i]].intensity - traj_copy->points[ind].intensity) > 20) || (abs(traj_copy->points[searchInd[i]].intensity - traj_copy->points[ind].intensity) <= 1)){
                        if((searchSqDis[i] <= surroundingKeyframeSearchRadius * 3) && (hiahia_if[searchInd[i]] != -1)){
                            toonearto_be1pose.push_back(searchInd[i]);   
                        }
                        else continue;
                    }
                    
                }

                if(toonearto_be1pose.size() != 0){
                    float sum_x = 0.0 + traj_copy->points[ind].x;
                    float sum_y = 0.0 + traj_copy->points[ind].y;

                    for(int j = 0; j < toonearto_be1pose.size(); j++){

                        sum_x += traj_copy->points[toonearto_be1pose[j]].x;
                        sum_y += traj_copy->points[toonearto_be1pose[j]].y;

                        // traj_copy->points[toonearto_be1pose[j]].x = 999999; // set onuse
                        // traj_copy->points[toonearto_be1pose[j]].y = 999999;
                        // traj_copy->points[toonearto_be1pose[j]].z = 999999;
                        hiahia_if[toonearto_be1pose[j]] = -1; // 更新为无法查找标志
                        pathsearch_tmp[toonearto_be1pose[j]] = ind; // 近邻设为同一地标
                    }

                    sum_x = sum_x / (toonearto_be1pose.size() + 1);  // normalize
                    sum_y = sum_y / (toonearto_be1pose.size() + 1);

                    pathcloud_noptr.points[pathInd].x = sum_x;
                    pathcloud_noptr.points[pathInd].y = sum_y;
                    pathcloud_noptr.points[pathInd].z = 0;
                    pathcloud_noptr.points[pathInd].intensity = pathInd;

                    // std::cout << "[Path Planner] " << "Keypath" << pathInd << ":" << " " << "[" << sum_x << ", " << sum_y << "]" << std::endl;

                    pathInd ++;

                }
            }

            else continue;
        }

        *pathCloud = pathcloud_noptr;

        // std::cout << "The num of KeyFrame is:" << " " << traj_copy->points.size() << "\n" 
        //           << "The num of KeyPose after normalization is:" << " " << (pathInd + 1) << std::endl;
        
    }
    
    /*
        功能：对于特定的两个位置搜索最短路径
    */
    void makePath(const pcl::PointCloud<PointType>::Ptr& cloudstrat, const pcl::PointCloud<PointType>::Ptr& cloudend){

        cloud2Relo(cloudstrat);
        cloud2Relo(cloudend);
        int start = pathsearch_tmp[relo_qureyIndex[0]];  // 定位起止索引
        int end = pathsearch_tmp[relo_qureyIndex[1]];

        // for(int i = 0; i < start; i++){  // remove start_front & end_back
        //     pathsearch_tmp.pop_front();
        // }
        // for(int j = pathsearch_tmp.size(); j > end; j--){
        //     pathsearch_tmp.pop_back();
        // }

        std::pair<int, int> start_end;
        // start_end = relo_newpath(start, end);


    }

    // std::pair<int, int> relo_newpath(int start, int end){
    //     std::pair<int, int> tmp;
    //     for(int i = 0; i < pathsearch.size(); i++){
    //         if(start == pathsearch[i]){
    //             tmp.first = i;
    //         }
    //         else if(end == pathsearch[i]){
    //             tmp.second = i;
    //         }
    //         else continue;
    //     }

    //     return tmp;
    // }

    // idea1
    // revisit whether or not
    int walkagain_ind(int cur, std::vector<int> pathwalked){
        if(pathwalked.size() != 0){
            for(int i = 0; i < pathwalked.size(); i++){
                if(cur == pathwalked[i]){
                    return pathwalked[i];
                }
                else return -1;
            }
        }
        else return -1;
    }

    // idea1
    // give back the index of place walked
    std::pair<std::vector<int>, std::vector<int>> walkagain(){

        std::pair<std::vector<int>, std::vector<int>> again_tmp;  // revisit place <start2end, end2start>
        std::vector<int> start2end; // the tmp vector of index
        std::vector<int> end2start;
        start2end.clear();
        end2start.clear();

        for(int i = 0; i < pathsearch_tmp.size(); i++){
            std::cout << pathsearch_tmp[i] << std::endl;
        }

        // start2end
        for(int id = 0; id < pathsearch_tmp.size(); id++){
            if(walkagain_ind(pathsearch_tmp[id], start2end) != -1){
                again_tmp.first.push_back(walkagain_ind(pathsearch_tmp[id], start2end));
                std::cout << again_tmp.first.back() << " ";

                start2end.clear(); // clear before
                start2end.push_back(pathsearch_tmp[id]); // current ind join the start2end as num.1
            }
            else{
                start2end.push_back(pathsearch_tmp[id]); // current ind just join the start2end at backend
            }
        }
        std::cout << 1 << std::endl;

        // end2start
        for(int id = pathsearch_tmp.size() - 1; id >= 0; id--){
            if(walkagain_ind(pathsearch_tmp[id], end2start) != -1){
                again_tmp.second.push_back(walkagain_ind(pathsearch_tmp[id], end2start));
                std::cout << again_tmp.second.back() << " ";

                end2start.clear();
                end2start.push_back(pathsearch_tmp[id]);
            }
            else{
                end2start.push_back(pathsearch_tmp[id]);
            }
        }
        std::cout << std::endl;

        return again_tmp;
        
    }

    // idea2
    void cut_path(){
        
        bool symbol = false;
        
        std::pair<int, std::vector<int>> path_part;  // one tmp part of path

        std::cout << pathsearch_tmp.size() << std::endl;  // show pathsearch_tmp
        for(int i = 0; i < pathsearch_tmp.size(); i++){
            std::cout << pathsearch_tmp[i] << std::endl;
        }

        for(int id = 0; id < pathsearch_tmp.size(); id++){
            if(id == 0){
                symbol = true;
                if_new_path.push_back(symbol);
                std::cout << if_new_path.back() << " ";
                path_part.first = id;
                path_part.second.push_back(pathsearch_tmp[id]);
            }
            else{
                if(abs(pathsearch_tmp[id] - pathsearch_tmp[id - 1]) < 60){
                    path_part.second.push_back(pathsearch_tmp[id]);
                }
                else{
                    all_path_part.push_back(path_part);
                    std::cout << all_path_part.back().first << std::endl;;

                    path_part.second.clear();
                    path_part.first = id;
                    path_part.second.push_back(pathsearch_tmp[id]);
                    

                    if(if_new_path.back() == true){
                        symbol = false;
                    }
                    else{
                        symbol = true;
                    }
                    if_new_path.push_back(symbol);
                    std::cout << if_new_path.back() << " ";
                    
                }
            }
        }

        // if(all_path_part.size() != if_new_path.size()){  // equal?
        //     ROS_ERROR("Path cut error !!");
        //     ros::shutdown();
        // }
        // else{
        //     std::cout << all_path_part.size() << std::endl;
        // }

        // for(int i = 0; i < all_path_part.size(); i++){
        //     std::cout << if_new_path[i] << std::endl;
        // }
        
    }    

    // show relo & pathplan
    void relocationShow(){

        SCDataBase(); // 生成SC数据库
        PathPlaner(); // 路径规划初始化

        // relocation from start to end 
        if(mode == "1"){
            
            int id = 5;
        
            ros::Rate rate(0.5);
            while(ros::ok()){
                if(id <= pcdFiles.size() - 5){
                    pcl::PointCloud<PointType>::Ptr thiscloud(new pcl::PointCloud<PointType>());
                    precloudName.push_back(pcdFiles[id]);
                    pcl::io::loadPCDFile(pcdFiles[id], *thiscloud);
                    std::cout << "[Relo Found]" << pcdFiles[id] << std::endl;
                    queryClouds.push_back(tmpClouds);
                    queryClouds.back().first = pcdFiles[id];
            
                    mtx.lock();
                    cloud2Relo(thiscloud);
                    mtx.unlock();

                    publishCloud(&publishReloCloud, thiscloud_final, ReloCloudMsg, "map");
                    if(publishReloPose.getNumSubscribers() != 0){
                        ReloPoseMsg.header.stamp = ros::Time().now();
                        ReloPoseMsg.header.frame_id = "map";
                        ReloPoseMsg.pose.pose.position.x = ReloPose.x;
                        ReloPoseMsg.pose.pose.position.y = ReloPose.y;
                        ReloPoseMsg.pose.pose.position.z = ReloPose.z;
                        publishReloPose.publish(ReloPoseMsg);
                    }
                
                    id += 2;

                    rate.sleep();
                }
                else{
                    // std::cout << "[Relo Info]" << "****** End the relocation ******" << endl;
                }
            }
            // for(int id = 120; id < 140; id++){ // relocation
            //     pcl::PointCloud<PointType>::Ptr thiscloud(new pcl::PointCloud<PointType>());
            //     precloudName.push_back(pcdFiles[id]);
            //     pcl::io::loadPCDFile(pcdFiles[id], *thiscloud);
            //     std::cout << "[Relo Found]" << pcdFiles[id] << std::endl;
            //     queryClouds.push_back(tmpClouds);
            //     queryClouds.back().first = pcdFiles[id];
            
            //     mtx.lock();
            //     cloud2Relo(thiscloud);
            //     mtx.unlock();
            
            // }

        }

        // plan the path just for certain two place
        else if(mode == "2"){
            // cut_path();
            // walkagain();
            ros::Rate rate(0.5);
            while(ros::ok()){
                publishCloud(&publishPath, pathCloud, PathMsg, "map"); // can open
            }
            
        }

    }

    // bool rankL2S(T a, T b){ // declare in unility.h
    //     return(fabs(a.real()) > fabs(b.real()));
    // }


    /*
        功能：重定位
    */
    void cloud2Relo(const pcl::PointCloud<PointType>::Ptr& thiscloud){
        
        /****   SC 粗配准    ****/
        curCloud2SC(thiscloud);

        auto cur_detectResult = scManagerRL.detectLoopClosureID();
        curLoopKeyPre.push_back(cur_detectResult.first);
        curYawDiffRad.push_back(cur_detectResult.second);

        // std::cout << scManagerRL.candidate_list.size() << std::endl;  // show
        // std::pair<std::string, std::vector<std::pair<int, double>>> tmpClouds; // "pre/query/yawdiff" to fill "queryCloud"
        
        std::pair<int, double> haha_0;
        haha_0.first = scManagerRL.candidate_list[0].first;
        haha_0.second = scManagerRL.candidate_list[0].second.second;
        queryClouds.back().second.push_back(haha_0);

        std::cout << "High confidence:" << " " << haha_0.first 
                                        << " " << scManagerRL.candidate_list[0].second.first
                                        << " " << haha_0.second
                                        << std::endl;

        for(int i = 1; i < scManagerRL.candidate_list.size(); i++){
            std::pair<int, double> haha_i;
            haha_i.first = scManagerRL.candidate_list[i].first;
            haha_i.second = scManagerRL.candidate_list[i].second.second;
            std::cout << "Low confidence:" << " " << haha_i.first 
                                           << " " << scManagerRL.candidate_list[i].second.first
                                           << " " << haha_i.second
                                           << std::endl;
            queryClouds.back().second.push_back(haha_i);
        }
        
        // std::cout << queryClouds.back().first << std::endl;  // test
        // for(int i = 0; i < queryClouds.back().second.size(); i++){
        //     std::cout <<  queryClouds.back().second[i].first << queryClouds.back().second[i].second << std::endl;
        // }

        scManagerRL.polarcontexts_.pop_back(); // 使用完后弹出
        scManagerRL.polarcontext_invkeys_.pop_back();
        scManagerRL.polarcontext_vkeys_.pop_back();
        scManagerRL.polarcontext_invkeys_mat_.pop_back();
        scManagerRL.candidate_list.clear(); // clear
    

        /****    submap + ICP 精匹配    ****/
        std::vector<std::pair<int, double> > ICPScore;
        std::vector<std::pair<int, Eigen::Affine3f>> icp_result_matrix;
        pcl::IterativeClosestPoint<PointType, PointType> icp;
        icp.setMaxCorrespondenceDistance(500);
        icp.setMaximumIterations(30);
        icp.setTransformationEpsilon(1e-6);
        icp.setEuclideanFitnessEpsilon(1e-6);
        icp.setRANSACIterations(0);

        pcl::PointCloud<PointType>::Ptr thiscloud_copy (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thiscloud_yawdiff (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thiscloud_yawdiff_trans (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr nouseCloud (new pcl::PointCloud<PointType>());

        for(int i = 0; i < queryClouds.back().second.size(); i++){
            Eigen::Affine3f icp_result_matrix_i;

            PointTypePose pose_yawdiff;
            PointTypePose pose_yawdiff_trans;
            thiscloud_copy->clear();
            thiscloud_yawdiff->clear();
            thiscloud_yawdiff_trans->clear();
            nouseCloud->clear();

            makeSubmap2ICP(queryClouds.back().second[i].first);  // submap

            pcl::copyPointCloud(*thiscloud, *thiscloud_copy);

            pose_yawdiff = transCur2ICP(queryClouds.back().second[i].second);
            *thiscloud_yawdiff += *transformCloud(thiscloud_copy, &pose_yawdiff);

            // pcl::copyPoint(KeyFramePose2ICP->points[queryClouds.back().second[i].first], pose_yawdiff_trans);
            pose_yawdiff_trans.x = KeyFramePose2ICP->points[queryClouds.back().second[i].first].x;
            pose_yawdiff_trans.y = KeyFramePose2ICP->points[queryClouds.back().second[i].first].y;
            pose_yawdiff_trans.z = KeyFramePose2ICP->points[queryClouds.back().second[i].first].z;
            pose_yawdiff_trans.roll = KeyFramePose2ICP->points[queryClouds.back().second[i].first].roll;
            pose_yawdiff_trans.pitch = KeyFramePose2ICP->points[queryClouds.back().second[i].first].pitch;
            pose_yawdiff_trans.yaw = KeyFramePose2ICP->points[queryClouds.back().second[i].first].yaw;


            *thiscloud_yawdiff_trans += *transformCloud(thiscloud_yawdiff, &pose_yawdiff_trans);

            icp.setInputSource(thiscloud_yawdiff_trans);
            icp.setInputTarget(subKeyframes);
            icp.align(*nouseCloud);

            if(icp.hasConverged() == false || icp.getFitnessScore() > historyKeyframeFitnessScore){
                continue;
            }
            else{
                std::pair<int, double> tmp_icp;
                tmp_icp.first = queryClouds.back().second[i].first;
                tmp_icp.second = icp.getFitnessScore();
                ICPScore.push_back(tmp_icp);

                std::pair<int, Eigen::Affine3f> tmp_result_matrix;
                Eigen::Affine3f icp_result_matrix_i;
                icp_result_matrix_i = icp.getFinalTransformation();; // trans finally
                tmp_result_matrix.first = queryClouds.back().second[i].first;
                tmp_result_matrix.second = icp_result_matrix_i;
                icp_result_matrix.push_back(tmp_result_matrix);

                
            }
         
        }
        if(ICPScore.size() != 0){
            std::sort(ICPScore.begin(), ICPScore.end(), rankS2L);

            relo_qureyIndex.push_back(ICPScore[0].first);  // 存储定位点云索引

            for(int i = 0; i < icp_result_matrix.size(); i++){
                if(icp_result_matrix[i].first == ICPScore[0].first){
                    pcl::PointCloud<PointType>::Ptr thiscloud_final_noDS (new pcl::PointCloud<PointType>());
                    PointTypePose query6D_i;
                    float icp_result[6];

                    ReloPose.x = KeyFramePose2ICP->points[ICPScore[0].first].x;
                    ReloPose.y = KeyFramePose2ICP->points[ICPScore[0].first].y;
                    ReloPose.z = KeyFramePose2ICP->points[ICPScore[0].first].z;
                    ReloPose.roll = KeyFramePose2ICP->points[ICPScore[0].first].roll;
                    ReloPose.pitch = KeyFramePose2ICP->points[ICPScore[0].first].pitch;
                    ReloPose.yaw = KeyFramePose2ICP->points[ICPScore[0].first].yaw;

                    pcl::getTranslationAndEulerAngles(icp_result_matrix[i].second, icp_result[0], icp_result[1], icp_result[2],
                                                                 icp_result[3], icp_result[4], icp_result[5]);
                    query6D_i.x = icp_result[0];
                    query6D_i.y = icp_result[1];
                    query6D_i.z = icp_result[2];
                    query6D_i.roll = icp_result[3];
                    query6D_i.pitch = icp_result[4];
                    query6D_i.yaw = icp_result[5];

                    thiscloud_final->clear();
                    thiscloud_final_noDS->clear();
                    *thiscloud_final_noDS += *transformCloud(thiscloud_yawdiff_trans, &query6D_i);
                    DownSample_relo.setInputCloud(thiscloud_final_noDS);
                    DownSample_relo.filter(*thiscloud_final);
                    thiscloud_final_noDS->clear();

                    break;   
                }
            }

            std::cout << "[Relo Best Choice]" << ICPScore[0].first << "\n"
                                              << "ICPFitnessScore is" << " " << ICPScore[0].second 
                                              << std::endl;
            // publish thiscloud_final && thiscloud
            std::cout << std::endl;
            
        }
        else{
            std::cout << "[Relo failed]" << "This place will be skipped .." << std::endl;
            std::cout << std::endl;
        }

        ICPScore.clear();
        icp_result_matrix.clear();
    }

    pcl::PointCloud<PointType>::Ptr transformCloud(const pcl::PointCloud<PointType>::Ptr& cloudIn, PointTypePose* transformIn){
        
        pcl::PointCloud<PointType>::Ptr cloudOut(new pcl::PointCloud<PointType>());
        PointType *pointFrom;

        int cloudSize = cloudIn->size();
        cloudOut->resize(cloudSize);

        // 得变换矩阵
        Eigen::Affine3f transCur = pcl::getTransformation(transformIn->x, transformIn->y, transformIn->z, transformIn->roll, transformIn->pitch, transformIn->yaw);
        
        #pragma omp parallel for num_threads(numberOfCores)
        for (int i = 0; i < cloudSize; ++i)
        {
            pointFrom = &cloudIn->points[i];
            cloudOut->points[i].x = transCur(0,0) * pointFrom->x + transCur(0,1) * pointFrom->y + transCur(0,2) * pointFrom->z + transCur(0,3);
            cloudOut->points[i].y = transCur(1,0) * pointFrom->x + transCur(1,1) * pointFrom->y + transCur(1,2) * pointFrom->z + transCur(1,3);
            cloudOut->points[i].z = transCur(2,0) * pointFrom->x + transCur(2,1) * pointFrom->y + transCur(2,2) * pointFrom->z + transCur(2,3);
            cloudOut->points[i].intensity = pointFrom->intensity;
        }
        return cloudOut;
    }
    

    // PCA for SC
    std::pair<Eigen::Vector2f, Eigen::Vector2f> pca4SC(const pcl::PointCloud<PointType>::ConstPtr& thisCloud){
        pcl::PointCloud<PointType>::Ptr cloud2pca (new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr cloud2pca_decenter (new pcl::PointCloud<PointType>());
        Eigen::Vector4f cloud2pca_center;
        Eigen::Vector3cf cloud2pca_3f;
        Eigen::Matrix3f cloud2pca_covariance_mat;
        
        Eigen::Vector2f cloud2pca_central;
        Eigen::Vector2f cloud2pca_2f;
        std::pair<Eigen::Vector2f, Eigen::Vector2f> pcaInfo;  // center/vector
        
        pcl::copyPointCloud(*thisCloud, *cloud2pca); // copy

        for(int i = 0; i < cloud2pca->points.size(); i++){
            cloud2pca->points[i].z = 0;  // project 2 axis z
        }

        pcl::compute3DCentroid<PointType>(*cloud2pca, cloud2pca_center);
        cloud2pca_central[0] = cloud2pca_center[0];
        cloud2pca_central[1] = cloud2pca_center[1];
        // std::cout << cloud2pca_central[0] << cloud2pca_central[1] << std::endl;
        pcl::computeCovarianceMatrix(*cloud2pca_decenter, cloud2pca_center, cloud2pca_covariance_mat);
        cloud2pca_covariance_mat.normalize();
        
        Eigen::EigenSolver<Eigen::Matrix3f> cloud2pca_solver(cloud2pca_covariance_mat);
        cloud2pca_3f = cloud2pca_solver.eigenvalues();

        auto hiahia = cloud2pca_3f[0].real();
        cloud2pca_2f[0] = cloud2pca_solver.eigenvectors()(0, 0).real();
        cloud2pca_2f[1] = cloud2pca_solver.eigenvectors()(1, 0).real();
        if(fabs(cloud2pca_3f[0].real()) > fabs(cloud2pca_3f[1].real())){
            if(fabs(cloud2pca_3f[0].real()) > fabs(cloud2pca_3f[2].real())){
                ;
            }
            else{
                hiahia = cloud2pca_3f[2].real();
                cloud2pca_2f[0] = cloud2pca_solver.eigenvectors()(0, 2).real();
                cloud2pca_2f[1] = cloud2pca_solver.eigenvectors()(1, 2).real();
            }
        }
        else{
            if(fabs(cloud2pca_3f[1].real()) > fabs(cloud2pca_3f[2].real())){
                hiahia = cloud2pca_3f[1].real();
                cloud2pca_2f[0] = cloud2pca_solver.eigenvectors()(0, 1).real();
                cloud2pca_2f[1] = cloud2pca_solver.eigenvectors()(1, 1).real();
            }
            else{
                hiahia = cloud2pca_3f[2].real();
                cloud2pca_2f[0] = cloud2pca_solver.eigenvectors()(0, 2).real();
                cloud2pca_2f[1] = cloud2pca_solver.eigenvectors()(1, 2).real();
            }
        }

        pcaInfo.first = cloud2pca_central;
        pcaInfo.second = cloud2pca_2f;
        return pcaInfo;
    }

    // get compensant_angle, get info of "b -a", if use tansformation, please use b -> a == -"b - a"
    std::pair<float, Eigen::Vector2f> getCompensant(const Eigen::Vector2f& a, const Eigen::Vector2f& b, const Eigen::Vector2f& a_center, const Eigen::Vector2f& b_center){
        float a_tan = atan2(a[1], a[0]);
        float b_tan = atan2(b[1], a[0]);
        float compensant_angle;
        Eigen::Vector2f compensant_xy;
        std::pair<float, Eigen::Vector2f> compensant;

        compensant_xy[0] = b_center[0] - a_center[0];
        compensant_xy[1] = b_center[1] - a_center[1];
        if(compensant_xy[0] < 0.0001) compensant_xy[0] = 0.00;
        if(compensant_xy[0] < 0.0001) compensant_xy[1] = 0.00;

        if(a_tan * b_tan >= 0){
            compensant_angle = - (b_tan - a_tan);
        }
        else{
            if((a_tan > 0) && (a[0] > 0)){
                if(b[0] > 0){
                    compensant_angle = - (b_tan - a_tan);
                }
                else{
                    compensant_angle = -(b_tan - a_tan) + PI;
                }
            }
            else if((a_tan > 0) && (a[0] < 0)){
                if(b[0] > 0){
                    compensant_angle = -(b_tan - a_tan) + PI;
                }
                else{
                    compensant_angle = - (b_tan - a_tan);
                }
            }
        }

        compensant.first = compensant_angle;
        compensant.second = compensant_xy;
        // std::cout << "[Compensant]" << "compensant_angle:" << " " << compensant.first << "\n"
        //                             << "compensant_xy:" << " " << compensant.second[0] << " " << compensant.second[1] << std::endl;
        return compensant;
    }

    // get tansformation for SC
    PointTypePose trans2KeyFrame0(const std::pair<float, Eigen::Vector2f>& compensant){
        PointTypePose pose2KeyFrame0;
        pose2KeyFrame0.x = - compensant.second[0];
        pose2KeyFrame0.y = - compensant.second[1];
        pose2KeyFrame0.z = 0;
        pose2KeyFrame0.roll = 0;
        pose2KeyFrame0.pitch = 0;
        pose2KeyFrame0.yaw = - compensant.first;
    }

    // trans for icp
    PointTypePose transCur2ICP(double yawdiff){
        PointTypePose poseCur2ICP;
        poseCur2ICP.x = poseCur2ICP.y = poseCur2ICP.z = 0;
        poseCur2ICP.roll = poseCur2ICP.pitch = 0;
        poseCur2ICP.yaw = - yawdiff; // right?
    }

    // submap2icp
    void makeSubmap2ICP(const int& key){
        subKeyframes->clear();
        int cloudSize = KeyFramePose2ICP->size();
        for(int i = -searchNum; i <= searchNum; i++){
            int keyNear = key + i;
            if (keyNear < 0 || keyNear >= cloudSize )
                continue;

            pcl::PointCloud<PointType>::Ptr hahaCloud(new pcl::PointCloud<PointType>());
            pcl::PointCloud<PointType>::Ptr hahaCloudDS(new pcl::PointCloud<PointType>());

            hahaCloud->clear();
            hahaCloudDS->clear();
            
            pcl::io::loadPCDFile(pcdFiles[keyNear], *hahaCloud);
            DownSample_sub.setInputCloud(hahaCloud);
            DownSample_sub.filter(*hahaCloudDS);

            *subKeyframes += *transformCloud(hahaCloud, &KeyFramePose2ICP->points[keyNear]);
        }
        // pcl::visualization::CloudViewer viewer("pcd viewer");
	    // viewer.showCloud(*subKeyframes);
    }

    double setPrecision(double value, int a = 3){
        double value_;
        if(abs(value) < 1){
            // *1000 + 0.5 取地板 /1000 == "保留3位有效数字"
            value_ = (floor(value * pow(10, a) + 0.5)) / pow(10, a);
        }
        else if((abs(value) >= 1) && (abs(value) < 10)){
            value_ = (floor(value * pow(10, a-1) + 0.5)) / pow(10, a-1);
        }
        else if((abs(value) >= 10) && (abs(value) < 100)){
            value_ = (floor(value * pow(10, a-2) + 0.5)) / pow(10, a-2);
        }
        else{
            value_ = (floor(value * pow(10, a-3) + 0.5)) / pow(10, a-3);
        }
        return value_;
    }

    void curCloud2SC(const pcl::PointCloud<PointType>::ConstPtr& thisrawCloud){
        pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame(new pcl::PointCloud<PointType>());
        pcl::PointCloud<PointType>::Ptr thisRawCloudKeyFrame20(new pcl::PointCloud<PointType>());
        pcl::copyPointCloud(*thisrawCloud,  *thisRawCloudKeyFrame); // copy

        // 随机yaw变换一次，不支持这样做，除非yaw旋转是bin大小的倍数
        thisRawCloudKeyFrame20->clear();
        *thisRawCloudKeyFrame20 += *transformCloud(thisRawCloudKeyFrame, &transformPCD); 
        
        // pcl::PointCloud<PointType>::Ptr pcdKeyFrame_0(new pcl::PointCloud<PointType>());  // tans curr 2 0
        // pcl::io::loadPCDFile(pcdFiles[0], *pcdKeyFrame_0);
        // std::pair<Eigen::Vector2f, Eigen::Vector2f> pcdInfo_0 = pca4SC(pcdKeyFrame_0);

        // std::pair<Eigen::Vector2f, Eigen::Vector2f> pcdInfo_i = pca4SC(thisRawCloudKeyFrame);
        // std::pair<float, Eigen::Vector2f> compensant = getCompensant(pcdInfo_0.first, pcdInfo_i.first, pcdInfo_0.second, pcdInfo_i.second);
        // PointTypePose transKeyFramei20 = trans2KeyFrame0(compensant);
        // *pcdKeyFrame_i_Aftrans += *transformCloud(pcdKeyFrame_i, &transKeyFramei20);
        // scManagerRL.makeAndSaveScancontextAndKeys(*pcdKeyFrame_i_Aftrans);

        Eigen::MatrixXd thisSC = scManagerRL.makeScancontext(*thisRawCloudKeyFrame20);
        // int precision = 3; // 保留三位有效数字，与建图时的保留位数统一，不然cos误差很大
        for(int i = 0; i < thisSC.rows(); i++){
            for(int j = 0; j < thisSC.cols(); j++){
                thisSC(i, j) = setPrecision(thisSC(i, j), 3);
            }
        }

        curSC.push_back(thisSC);
        curRingkey.push_back(scManagerRL.makeRingkeyFromScancontext(curSC.back()));
        curSectorkey.push_back(scManagerRL.makeSectorkeyFromScancontext(curSC.back()));
        cur_polarcontext_invkeys_mat.push_back(eig2stdvec(curRingkey.back()));

        // for(int i = 0; i < thisSC.rows(); i++){ // show
        //     for(int j = 0; j < thisSC.cols(); j++){
        //         std::cout << thisSC(i, j)<< " ";
        //     }
        //     std::cout << std::endl;
        // }

        // 暂时将curSC加入scManagerRL的数据库，后面在curSC重定位结束后会弹出
        scManagerRL.polarcontexts_.push_back(curSC.back()); 
        scManagerRL.polarcontext_invkeys_.push_back(curRingkey.back());
        scManagerRL.polarcontext_vkeys_.push_back(curSectorkey.back());
        scManagerRL.polarcontext_invkeys_mat_.push_back(cur_polarcontext_invkeys_mat.back());
        
        thisRawCloudKeyFrame->clear();
    }

    void SCDataBase(){
        
        if(pcdFiles.size() < scManagerRL.NUM_EXCLUDE_RECENT){
            std::cout << "[Relo Error]" << "Not enough SCDB !!" << std::endl;
            return ;
        }

        // // random cloud to relocation using aompensant with high robust !! 
        // pcl::PointCloud<PointType>::Ptr pcdKeyFrame_0(new pcl::PointCloud<PointType>());
        // pcl::io::loadPCDFile(pcdFiles[0], *pcdKeyFrame_0);
        // std::pair<Eigen::Vector2f, Eigen::Vector2f> pcdInfo_0 = pca4SC(pcdKeyFrame_0);
        
        // for(int i = 700; i < 701; i++){
        //     pcl::PointCloud<PointType>::Ptr pcdKeyFrame_i(new pcl::PointCloud<PointType>());
        //     pcl::PointCloud<PointType>::Ptr pcdKeyFrame_i_Aftrans(new pcl::PointCloud<PointType>());
        //     pcl::io::loadPCDFile(pcdFiles[i], *pcdKeyFrame_i);
            
        //     std::pair<Eigen::Vector2f, Eigen::Vector2f> pcdInfo_i = pca4SC(pcdKeyFrame_i);
        //     std::pair<float, Eigen::Vector2f> compensant = getCompensant(pcdInfo_0.first, pcdInfo_i.first, pcdInfo_0.second, pcdInfo_i.second);
        //     PointTypePose transKeyFramei20 = trans2KeyFrame0(compensant);
        //     *pcdKeyFrame_i_Aftrans += *transformCloud(pcdKeyFrame_i, &transKeyFramei20);
        //     scManagerRL.makeAndSaveScancontextAndKeys(*pcdKeyFrame_i_Aftrans);

        //     // ReloCloud_result->clear(); // show the result of compensant
        //     // *ReloCloud_result += *pcdKeyFrame_i_Aftrans;

        // }

        // please read your whole database during slam, i ~ scdFiles.size()
        for(int i = 0; i < scdFiles.size(); i++){

            pcl::PointCloud<PointType>::Ptr pcdCloud(new pcl::PointCloud<PointType>());
            pcl::io::loadPCDFile(pcdFiles[i], *pcdCloud);
            dataBaseVector.push_back(pca4SC(pcdCloud)); // make database vector

            std::string line;
            std::vector<std::string> line2eigen; // 转Eigen::MatrixXd SC
            Eigen::MatrixXd SC_tmp = MatrixXd::Ones(scManagerRL.PC_NUM_RING, scManagerRL.PC_NUM_SECTOR);
        
            SCD_File.open(scdFiles[i]);
            int row = 0;
            while(getline(SCD_File, line)){
                line2eigen.clear();               
                boost::split(line2eigen, line, boost::is_any_of(" "));
                // std::cout << line2eigen.size() << std::endl;
                for(int col = 0; col < line2eigen.size(); col++){
                    SC_tmp(row, col) = atof(line2eigen[col].c_str());    
                    // std::cout << SC_tmp(row, col) << " ";    
                }
                // std::cout << std::endl;
                row ++;
            }
            SCD_File.close();

            // for(int i = 0; i < SC_tmp.rows(); i++){  // show
            //     for(int j = 0; j < SC_tmp.cols(); j++){
            //         std::cout << SC_tmp(i, j)<< " ";
            //     }
            //     std::cout << std::endl;
            // }

            // showSC(SC_tmp);
            
            scManagerRL.polarcontexts_.push_back(SC_tmp);
            scManagerRL.polarcontext_invkeys_.push_back(scManagerRL.makeRingkeyFromScancontext(SC_tmp));
            scManagerRL.polarcontext_vkeys_.push_back(scManagerRL.makeSectorkeyFromScancontext(SC_tmp));
            
            std::vector<float> polarcontext_invkey_vec = eig2stdvec(scManagerRL.makeRingkeyFromScancontext(SC_tmp));
            scManagerRL.polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);
        }

        for(int i = 0; i < scManagerRL.NUM_EXCLUDE_RECENT; i++){
            Eigen::MatrixXd SC_nouse = MatrixXd::Ones(scManagerRL.PC_NUM_RING, scManagerRL.PC_NUM_SECTOR);

            for(int i = 0; i < SC_nouse.rows(); i++){  // make nouse
                for(int j = 0; j < SC_nouse.cols(); j++){
                    SC_nouse(i, j) = 10000;
                }
            }

            scManagerRL.polarcontexts_.push_back(SC_nouse);
            scManagerRL.polarcontext_invkeys_.push_back(scManagerRL.makeRingkeyFromScancontext(SC_nouse));
            scManagerRL.polarcontext_vkeys_.push_back(scManagerRL.makeSectorkeyFromScancontext(SC_nouse));
            
            std::vector<float> polarcontext_invkey_vec = eig2stdvec(scManagerRL.makeRingkeyFromScancontext(SC_nouse));
            scManagerRL.polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);

        }

    //     std::cout << "[SC Database]" << "The scale of SCDB is " << scManagerRL.polarcontexts_.size() - scManagerRL.NUM_EXCLUDE_RECENT << "," 
    //                                                             << scManagerRL.polarcontext_invkeys_.size() - scManagerRL.NUM_EXCLUDE_RECENT<< "," 
    //                                                             << scManagerRL.polarcontext_vkeys_.size() - scManagerRL.NUM_EXCLUDE_RECENT << ","
    //                                                             << scManagerRL.polarcontext_invkeys_mat_.size() - scManagerRL.NUM_EXCLUDE_RECENT
    //                                                             << std::endl;
    }

    void showSC(const Eigen::MatrixXd& SC2Show){
        cv::eigen2cv(SC2Show, SCImage);
        SCImage.convertTo(SCImage, CV_32FC1);
        SCImage *= 1000;
        cv::imshow("sc", SCImage);
        cv::waitKey(0);
    }
};


int main(int argc, char **argv){

    ros::init(argc, argv, "lio_sam");

    Relocation RL;

    // if(argc != 4){
    //     ROS_ERROR("The num of parameters given is not right !!");
    //     ros::shutdown();
    // }
    // RL.mode = argv[3];

    RL.mode = "1";

    ROS_INFO("\033[1;32m----> Relocation Started.\033[0m");


    // std::thread在各个发布中只设置rate和rate.sleep，最后一个总的ros::spin()
    std::thread cloudpubilsh(&Relocation::publishWorld, &RL);
    std::thread reloshow(&Relocation::relocationShow, &RL);
    
    ros::spin();  

    cloudpubilsh.join();
    reloshow.join();

    // RL.PathPlaner();

    // RL.SCDataBase();
    // RL.publishWorld();
    // RL.relocationShow();
    // RL.makeSubmap2ICP(100);
    // RL.publishWorld();

    return 0;
}