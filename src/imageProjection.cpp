#include "utility.h"
#include "lio_sam/cloud_info.h"

// 定义了针对不同激光雷达类型或数据集类型的点云格式
struct VelodynePointXYZIRT
{
    PCL_ADD_POINT4D
    PCL_ADD_INTENSITY;
    uint16_t ring;
    float time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT (VelodynePointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint16_t, ring, ring) (float, time, time)
)

struct OusterPointXYZIRT {
    PCL_ADD_POINT4D;
    float intensity;
    uint32_t t;
    uint16_t reflectivity;
    uint8_t ring;
    uint16_t noise;
    uint32_t range;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;
POINT_CLOUD_REGISTER_POINT_STRUCT(OusterPointXYZIRT,
    (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
    (uint32_t, t, t) (uint16_t, reflectivity, reflectivity)
    (uint8_t, ring, ring) (uint16_t, noise, noise) (uint32_t, range, range)
)

struct MulranPointXYZIRT { // from the file player's topic https://github.com/irapkaist/file_player_mulran, see https://github.com/irapkaist/file_player_mulran/blob/17da0cb6ef66b4971ec943ab8d234aa25da33e7e/src/ROSThread.cpp#L7
     PCL_ADD_POINT4D;
     float intensity;
     uint32_t t;
     int ring;

     EIGEN_MAKE_ALIGNED_OPERATOR_NEW
 }EIGEN_ALIGN16;
 POINT_CLOUD_REGISTER_POINT_STRUCT (MulranPointXYZIRT,
     (float, x, x) (float, y, y) (float, z, z) (float, intensity, intensity)
     (uint32_t, t, t) (int, ring, ring)
 )

// Use the Velodyne point format as a common representation
using PointXYZIRT = VelodynePointXYZIRT;

const int queueLength = 2000; // 队列长度

class ImageProjection : public ParamServer
{
private:

    std::mutex imuLock; // 定义两个锁
    std::mutex odoLock;

    ros::Subscriber subLaserCloud;
    ros::Publisher  pubLaserCloud;
    
    ros::Publisher pubExtractedCloud;
    ros::Publisher pubLaserCloudInfo;

    ros::Subscriber subImu;
    std::deque<sensor_msgs::Imu> imuQueue;

    ros::Subscriber subOdom;
    std::deque<nav_msgs::Odometry> odomQueue;

    std::deque<sensor_msgs::PointCloud2> cloudQueue;
    sensor_msgs::PointCloud2 currentCloudMsg;

    double *imuTime = new double[queueLength]; // imu动态分配
    double *imuRotX = new double[queueLength];
    double *imuRotY = new double[queueLength];
    double *imuRotZ = new double[queueLength];

    int imuPointerCur;
    bool firstPointFlag;
    Eigen::Affine3f transStartInverse;

    pcl::PointCloud<PointXYZIRT>::Ptr laserCloudIn;
    pcl::PointCloud<OusterPointXYZIRT>::Ptr tmpOusterCloudIn;
    pcl::PointCloud<MulranPointXYZIRT>::Ptr tmpMulranCloudIn;
    pcl::PointCloud<PointType>::Ptr   fullCloud;
    pcl::PointCloud<PointType>::Ptr   extractedCloud;

    int deskewFlag; // 通道检查标志
    cv::Mat rangeMat; // range image

    bool odomDeskewFlag; // 去畸变标志
    float odomIncreX;
    float odomIncreY;
    float odomIncreZ;

    lio_sam::cloud_info cloudInfo; // cloud_info这是什么意思？是自定义的msg文件，每一帧的激光点云
    double timeScanCur;
    double timeScanEnd;
    std_msgs::Header cloudHeader;


public:
    ImageProjection():
    deskewFlag(0)
    {
        // “class内使用对象实例化”进行订阅话题与回调函数格式：
        // sub* = nh.subscibe<消息类型>(话题名称， 队列块大小， 回调函数， this， 传输形式)
        // 回调函数是类内函数，是引用的格式；this是指对象实例化后，调用自身的回调函数；ros::TransportHints().tcpNoDelay()是无延迟传输方式
        // 队列块大小是一个缓存大小，防止消息堵塞，但是回调函数本身每次只能处理一个数据，
        
        // 订阅imu原始数据
        subImu        = nh.subscribe<sensor_msgs::Imu>(imuTopic, 2000, &ImageProjection::imuHandler, this, ros::TransportHints().tcpNoDelay());
        
        // 订阅imu增量式里程计：来自IMUPreintegration发布的增量式里程计话题(前一帧激光帧优化基础上)
        subOdom       = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental", 2000, &ImageProjection::odometryHandler, this, ros::TransportHints().tcpNoDelay());
        
        // 定义原始激光点云
        subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>(pointCloudTopic, 5, &ImageProjection::cloudHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布去畸变点云，pub* = nh.advertise<消息类型>(话题名称， 队列长度(点云必然是一帧一帧发布的))
        pubExtractedCloud = nh.advertise<sensor_msgs::PointCloud2> ("lio_sam/deskew/cloud_deskewed", 1);
        
        // 发布激光点云信息，发布类型的自定义的cloud_msg格式
        pubLaserCloudInfo = nh.advertise<lio_sam::cloud_info> ("lio_sam/deskew/cloud_info", 1);

        allocateMemory(); // 分配内存
        resetParameters(); // 重置部分参数

        // setVerbosityLevel用于设置控制台输出的信息
        // L_ALWAYS不输出任何消息；L_DEBUG会输出DEBUG信息;L_ERROR会输出ERROR信息
        pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
    }

    void allocateMemory()
    {
        // 根据params.yaml中给出的N_SCAN(行)、Horizon_SCAN(列)参数值分配内存
        // 使用动态指针的.reset()初始化
        laserCloudIn.reset(new pcl::PointCloud<PointXYZIRT>());
        tmpOusterCloudIn.reset(new pcl::PointCloud<OusterPointXYZIRT>());
        tmpMulranCloudIn.reset(new pcl::PointCloud<MulranPointXYZIRT>());
        fullCloud.reset(new pcl::PointCloud<PointType>());
        extractedCloud.reset(new pcl::PointCloud<PointType>());

        fullCloud->points.resize(N_SCAN*Horizon_SCAN);

        // cloudinfo是msg文件下自定义的cloud_info消息，对其中的变量进行赋值操作
        cloudInfo.startRingIndex.assign(N_SCAN, 0); // 索引
        cloudInfo.endRingIndex.assign(N_SCAN, 0);

        cloudInfo.pointColInd.assign(N_SCAN*Horizon_SCAN, 0); // 一维表示形式
        cloudInfo.pointRange.assign(N_SCAN*Horizon_SCAN, 0);

        resetParameters(); // 重置部分参数
    }

    // declare for imu_guess_poses record
    std::vector<tf::Matrix3x3> imu_guess_rotation;
    std::vector<tf::Vector3>  imu_guess_translation;
    const std::string imu_initial_guess_pose {savePCDDirectory + "imuGuess_poses.txt"}; // imuguess record记录
    
    void resetParameters()
    {
        laserCloudIn->clear(); // 清零
        extractedCloud->clear();

        // reset range matrix for range image projection
        rangeMat = cv::Mat(N_SCAN, Horizon_SCAN, CV_32F, cv::Scalar::all(FLT_MAX)); // 初始全部用FLT_MAX 填充

        imuPointerCur = 0;
        firstPointFlag = true; // 第一帧标志
        odomDeskewFlag = false; // 去畸变标志

        for (int i = 0; i < queueLength; ++i)
        {
            imuTime[i] = 0;
            imuRotX[i] = 0;
            imuRotY[i] = 0;
            imuRotZ[i] = 0;
        }
    }

    ~ImageProjection(){} // 析构？？析构了main()函数里的实例对象不就没了？？但是直接析构对象可以确保每一激光帧互不干扰！！！

    // TODO: time record
    void record_time(const std::string & file_name, const std::vector<double> time){
        std::fstream stream(file_name.c_str(), fstream::out); // cin txt
        for(int i = 0; i < time.size(); i++){
            stream << std::to_string(i - 1) << " " << time[i] << std::endl;
        }
    }

    /*
        功能：记录cloud_info中的initialguess值，保存为txt  // fyx
        TODO： 记录imu估计位姿，感觉记录下的初始位姿不太准
    */
    void record_initialguesss(std::string _filename){

        std::fstream stream(_filename.c_str(), fstream::out); // cin txt
       
        // tf::Quaternion imu_quaternion;
        // tf::Vector3 imu_translation;

        // imu_quaternion.setRPY(cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
        // imu_translation.setX(cloudInfo.initialGuessX);
        // imu_translation.setY(cloudInfo.initialGuessY);
        // imu_translation.setZ(cloudInfo.initialGuessZ);

        // tf::Matrix3x3 rotation = tf::Matrix3x3(imu_quaternion); // rotation matrix
        for(int i = 0; i < imu_guess_rotation.size(); i++){
            stream << imu_guess_rotation[i][0][0] << " " << imu_guess_rotation[i][0][1] << " " << imu_guess_rotation[i][0][2] << " " << imu_guess_translation[i].getX() << " "
                   << imu_guess_rotation[i][1][0] << " " << imu_guess_rotation[i][1][1] << " " << imu_guess_rotation[i][1][2] << " " << imu_guess_translation[i].getY() << " " 
                   << imu_guess_rotation[i][2][0] << " " << imu_guess_rotation[i][2][1] << " " << imu_guess_rotation[i][2][2] << " " << imu_guess_translation[i].getZ() << std::endl;
        }
        // auto col1 = rotation[0][0];
        
   }
    /*
        功能：订阅Imu原始数据
        同时，将imu数据转换到“中间系”(与lidar系相差一个平移)
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imuMsg)
    {
        sensor_msgs::Imu thisImu = imuConverter(*imuMsg); // 转“中间系”

        std::lock_guard<std::mutex> lock1(imuLock);
        imuQueue.push_back(thisImu);

        // debug IMU data
        // cout << std::setprecision(6);
        // cout << "IMU acc: " << endl;
        // cout << "x: " << thisImu.linear_acceleration.x << 
        //       ", y: " << thisImu.linear_acceleration.y << 
        //       ", z: " << thisImu.linear_acceleration.z << endl;
        // cout << "IMU gyro: " << endl;
        // cout << "x: " << thisImu.angular_velocity.x << 
        //       ", y: " << thisImu.angular_velocity.y << 
        //       ", z: " << thisImu.angular_velocity.z << endl;
        // double imuRoll, imuPitch, imuYaw;
        // tf::Quaternion orientation;
        // tf::quaternionMsgToTF(thisImu.orientation, orientation);
        // tf::Matrix3x3(orientation).getRPY(imuRoll, imuPitch, imuYaw);
        // cout << "IMU roll pitch yaw: " << endl;
        // cout << "roll: " << imuRoll << ", pitch: " << imuPitch << ", yaw: " << imuYaw << endl << endl;
    }

    /*
        功能：订阅Imu增量式里程计回调函数
        imu的增量式里程计是imuPreintegration在上一帧isam2优化基础上，预积分计算得到的每时刻imu的增量位姿
    */
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odometryMsg)
    {
        std::lock_guard<std::mutex> lock2(odoLock);
        odomQueue.push_back(*odometryMsg);
    }

    /*
        功能：订阅点云回调函数
    */ 
    void cloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // 添加一帧激光点云到队列，取出最早一帧作为当前帧，计算起止时间戳，检查数据有效性
        if (!cachePointCloud(laserCloudMsg))
            return;

        // Imu和点云去畸变
        if (!deskewInfo())
            return;

        // 投影至range image
        projectPointCloud();

        // 提取有效点云，存储在extractedCloud
        cloudExtraction();

        // 发布当前帧校正好的点云
        publishClouds();

        // 重置参数，接受每一帧lidar数据都要重置这些参数
        resetParameters();
    }

    /*
        功能：缓存3帧帧点云，检查点云数据的有效性
        检查是否存在ring和time通道，更针对传感器测量数据格式进行检查
    */
    bool cachePointCloud(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
    {
        // cache point cloud
        cloudQueue.push_back(*laserCloudMsg);
        if (cloudQueue.size() <= 2) // 至少存3帧点云
            return false;

        // convert cloud
        // 取出缓存点云队列的最早帧为当前帧，并弹出
        currentCloudMsg = std::move(cloudQueue.front());

        cloudHeader = currentCloudMsg.header;

        cloudQueue.pop_front();

        // 判断传感器类型，但最终都要转换为“velodyne格式”
        if (sensor == SensorType::VELODYNE)
        {
            pcl::moveFromROSMsg(currentCloudMsg, *laserCloudIn);
        }

        // 下面是将任意SensorType转换为velodyne格式，感觉没太大变化
        else if (sensor == SensorType::OUSTER)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpOusterCloudIn);
            laserCloudIn->points.resize(tmpOusterCloudIn->size());
            laserCloudIn->is_dense = tmpOusterCloudIn->is_dense;
            for (size_t i = 0; i < tmpOusterCloudIn->size(); i++)
            {
                auto &src = tmpOusterCloudIn->points[i]; // 引用防止内存消耗
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = src.t * 1e-9f;
            }
        }
        else if (sensor == SensorType::MULRAN)
        {
            // Convert to Velodyne format
            pcl::moveFromROSMsg(currentCloudMsg, *tmpMulranCloudIn);
            laserCloudIn->points.resize(tmpMulranCloudIn->size());
            laserCloudIn->is_dense = tmpMulranCloudIn->is_dense;
            for (size_t i = 0; i < tmpMulranCloudIn->size(); i++)
            {
                auto &src = tmpMulranCloudIn->points[i];
                auto &dst = laserCloudIn->points[i];
                dst.x = src.x;
                dst.y = src.y;
                dst.z = src.z;
                dst.intensity = src.intensity;
                dst.ring = src.ring;
                dst.time = float(src.t);
            }
        }
        else
        {
            ROS_ERROR_STREAM("Unknown sensor type: " << int(sensor));
            ros::shutdown();
        }

        // get timestamp
        // 这一帧点云的时间戳被记录下来，存入timeScanCur中，函数deskewPoint中会被加上laserCloudIn->points[i].time
        timeScanCur = cloudHeader.stamp.toSec();

        // 当前帧点云的截止时间
        timeScanEnd = timeScanCur + laserCloudIn->points.back().time;

        // check dense flag
        if (laserCloudIn->is_dense == false) // is_dense == true是指点全部有序
        {
            ROS_ERROR("Point cloud is not in dense format, please remove NaN points first!");
            ros::shutdown();
        }

        // check ring channel
        static int ringFlag = 0; // static关键字，只检查一次
        // 检查ring这个field是否存在. veloodyne和ouster都有，如果没有的话就要进行几何计算来确定ring
        // ring代表线数，0是最下面那条
        if (ringFlag == 0)
        {
            ringFlag = -1;
            for (int i = 0; i < (int)currentCloudMsg.fields.size(); ++i)
            {
                if (currentCloudMsg.fields[i].name == "ring")
                {
                    ringFlag = 1;
                    break;
                }
            }
            if (ringFlag == -1)
            {
                ROS_ERROR("Point cloud ring channel not available, please configure your point cloud data!");
                ros::shutdown();
            }
        }

        // check point time
        // 检查是否存在time通道
        if (deskewFlag == 0)
        {
            deskewFlag = -1;
            for (auto &field : currentCloudMsg.fields)
            {
                if (field.name == "time" || field.name == "t")
                {
                    deskewFlag = 1;
                    break;
                }
            }
            if (deskewFlag == -1)
                ROS_WARN("Point cloud timestamp not available, deskew function disabled, system will drift significantly!");
        }

        return true;
    }

    /*
        功能：总的去畸变处理(Imu和激光点云)
        检测当前Imu数据是否可用：一是需要原始imu队列不为空；二是imu的时间段应该覆盖当前帧扫描时间段
    */ 
    bool deskewInfo()
    {
        // 与imu和odo回调函数一起使用这两个锁，形成互斥
        // 如果当前互斥量被其他线程锁住，则当前的调用线程被阻塞住！！
        std::lock_guard<std::mutex> lock1(imuLock); 
        std::lock_guard<std::mutex> lock2(odoLock);

        // make sure IMU data available for the scan
        // 确保Imu数据覆盖前后两个激光帧
        if (imuQueue.empty() || imuQueue.front().header.stamp.toSec() > timeScanCur || imuQueue.back().header.stamp.toSec() < timeScanEnd)
        {
            ROS_DEBUG("Waiting for IMU data ...");
            return false;
        }

        // 当前点云帧的原始imu数据处理，计算旋转畸变参数
        // 1、遍历当前激光帧起止时刻之间的imu数据(此时imu数据的时间段是覆盖整个激光帧的扫描时间段的)，初始时刻对应的imu姿态rpy设为当前激光帧的初始姿态角
        // 2、用角速度、时间积分，计算“每一时刻imu相对于初始时刻imu的旋转量“(imu预积分本质思想)，初始时刻对应旋转量为0
        // 注：此时的imu数据已经转到“中间系”(与lidar系差个平移)！！
        imuDeskewInfo(); 
        
        // imu里程计用于计算平移畸变参数
        // 以当前激光帧时刻的Imu的姿态rpy为初值，计算起止每一时刻imu相对于初始时刻总的姿态变换rpy和位移xyz
        odomDeskewInfo(); 

        return true;
    }

    /*
        功能：计算imu去畸变数据
    */
    void imuDeskewInfo()
    {
        cloudInfo.imuAvailable = false;

        // 从原始Imu队列(已变换到“中间系”)删除当前激光帧0.01s前面的imu数据，imu的频率好像是50Hz
        while (!imuQueue.empty())
        {
            if (imuQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                imuQueue.pop_front();
            else
                break;
        }

        if (imuQueue.empty())
            return;

        imuPointerCur = 0; // imu数据处理次数计数

        for (int i = 0; i < (int)imuQueue.size(); ++i) // 遍历所有imu数据
        {
            sensor_msgs::Imu thisImuMsg = imuQueue[i];
            double currentImuTime = thisImuMsg.header.stamp.toSec();

            // get roll, pitch, and yaw estimation for this scan
            // thisImuMsg内提取姿态角rpy
            if (currentImuTime <= timeScanCur)
                imuRPY2rosRPY(&thisImuMsg, &cloudInfo.imuRollInit, &cloudInfo.imuPitchInit, &cloudInfo.imuYawInit); // 读取thisImuMsg中的rpy

            if (currentImuTime > timeScanEnd + 0.01) // 计数到激光帧扫描结束的0.01s
                break;

            if (imuPointerCur == 0){ // 第一帧旋转增量初始化为0
                imuRotX[0] = 0;
                imuRotY[0] = 0;
                imuRotZ[0] = 0;
                imuTime[0] = currentImuTime;
                ++imuPointerCur;
                continue;
            }

            // get angular velocity
            // thisImuMsg提取角速度
            double angular_x, angular_y, angular_z;
            imuAngular2rosAngular(&thisImuMsg, &angular_x, &angular_y, &angular_z);

            // integrate rotation，这跟用gtsam的预积分有区别吗？？为什么不用？？
            // 积分求旋转增量，imuRotX中的一个元素是上一个元素加上旋转增量
            double timeDiff = currentImuTime - imuTime[imuPointerCur-1];
            imuRotX[imuPointerCur] = imuRotX[imuPointerCur-1] + angular_x * timeDiff;
            imuRotY[imuPointerCur] = imuRotY[imuPointerCur-1] + angular_y * timeDiff;
            imuRotZ[imuPointerCur] = imuRotZ[imuPointerCur-1] + angular_z * timeDiff;
            imuTime[imuPointerCur] = currentImuTime;
            ++imuPointerCur;
        }

        --imuPointerCur; // 多加了一个1？？没有吧？？啥意思

        if (imuPointerCur <= 0) // 无合规的imu数据
            return;

        cloudInfo.imuAvailable = true; // imu可用标志
    }

    /*
        功能：去畸变IMUPreintegration发布的imu增量式里程计话题(地图优化后的)
        初始pose信息保存在cloudInfo里
        其实这就是对点云去畸变？？还没开始呢，还是在准备参数！！

            imuDeskewInfo()利用imu原始数据，以当前激光帧初始时刻imu旋转为0，扫描时间段内每时刻imu相对于初始时刻的旋转量
            odomDeskewInfo()
    */ 
    void odomDeskewInfo()
    {
        cloudInfo.odomAvailable = false;

        // 从imu增量里程计队列中删除当前激光帧0.01s前面时刻的imu数据
        while (!odomQueue.empty())
        {
            if (odomQueue.front().header.stamp.toSec() < timeScanCur - 0.01)
                odomQueue.pop_front();
            else
                break;
        }

        if (odomQueue.empty())
            return;

        if (odomQueue.front().header.stamp.toSec() > timeScanCur) // 无当前激光帧之前的里程计则返回
            return;

        // get start odometry at the beinning of the scan
        nav_msgs::Odometry startOdomMsg;

        // 通过遍历得到满足时间戳要求的第一个超出当前激光帧扫描开始时间的imu增量里程计数据，第一个可用的imu增量里程计
        for (int i = 0; i < (int)odomQueue.size(); ++i) 
        {
            startOdomMsg = odomQueue[i];

            if (ROS_TIME(&startOdomMsg) < timeScanCur)
                continue;
            else
                break;
        }

        // 提取初始Imu里程计的姿态角，为啥非要用tf中的变换函数？？图个方便或精确？难道时刻被监听着？？
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(startOdomMsg.pose.pose.orientation, orientation);

        double roll, pitch, yaw;
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);

        // Initial guess used in mapOptimization
        // 用当前激光帧起始时刻的imu里程计，初始化Lidar位姿，后面用于mapOptimization
        cloudInfo.initialGuessX = startOdomMsg.pose.pose.position.x;
        cloudInfo.initialGuessY = startOdomMsg.pose.pose.position.y;
        cloudInfo.initialGuessZ = startOdomMsg.pose.pose.position.z;
        cloudInfo.initialGuessRoll  = roll;
        cloudInfo.initialGuessPitch = pitch;
        cloudInfo.initialGuessYaw   = yaw;

        tf::Quaternion imu_quaternion; // fyx
        tf::Vector3 imu_translation;

        imu_quaternion.setRPY(cloudInfo.initialGuessRoll, cloudInfo.initialGuessPitch, cloudInfo.initialGuessYaw);
        imu_translation.setX(cloudInfo.initialGuessX);
        imu_translation.setY(cloudInfo.initialGuessY);
        imu_translation.setZ(cloudInfo.initialGuessZ);

        tf::Matrix3x3 rotation = tf::Matrix3x3(imu_quaternion); // rotation matrix

        // record_initialguesss(imu_initial_guess_pose); // fyx, record imu_guess_pose for lidar scan2map
        imu_guess_rotation.push_back(rotation);
        imu_guess_translation.push_back(imu_translation);
        
        // show for test
        // std::cout << imu_guess_rotation.size() << " " << imu_guess_translation.size() << std::endl; // fyx
        
        cloudInfo.odomAvailable = true; // valid odometery flag

        // get end odometry at the end of the scan
        odomDeskewFlag = false;

        if (odomQueue.back().header.stamp.toSec() < timeScanEnd) // imu里程计数据太短，没有覆盖当前激光帧的整个扫描阶段则返回
            return;

        nav_msgs::Odometry endOdomMsg;

        // 通过遍历找到imu里程计找最早超过当前激光帧扫描结束时刻的数据，最后一个可用的imu增量里程计
        for (int i = 0; i < (int)odomQueue.size(); ++i)
        {
            endOdomMsg = odomQueue[i];

            if (ROS_TIME(&endOdomMsg) < timeScanEnd)
                continue;
            else
                break;
        }

        // 如果起止时刻对应的imu里程计协方差不等，则返回？？这是什么意思？？首元素.pose.covariance[0]到底是什么？
        if (int(round(startOdomMsg.pose.covariance[0])) != int(round(endOdomMsg.pose.covariance[0])))
            return;

        // 得到起止时刻各自的imu增量里程计的位姿
        Eigen::Affine3f transBegin = pcl::getTransformation(startOdomMsg.pose.pose.position.x, startOdomMsg.pose.pose.position.y, startOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        tf::quaternionMsgToTF(endOdomMsg.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        Eigen::Affine3f transEnd = pcl::getTransformation(endOdomMsg.pose.pose.position.x, endOdomMsg.pose.pose.position.y, endOdomMsg.pose.pose.position.z, roll, pitch, yaw);

        // transBegin.inverse()取逆(为啥取逆你懂的^_^)再乘transEnd，得到最开始Imu增量里程计位姿和最后Imu增量里程计位姿之间的相对位姿
        Eigen::Affine3f transBt = transBegin.inverse() * transEnd;

        // 相对变换，提取平移增量、旋转增量rpy
        float rollIncre, pitchIncre, yawIncre;

        // 在给定起始Imu增量位姿之间的相对变换中提取xyz和rpy，用于后续激光点云去畸变
        pcl::getTranslationAndEulerAngles(transBt, odomIncreX, odomIncreY, odomIncreZ, rollIncre, pitchIncre, yawIncre);

        odomDeskewFlag = true; // succeed
    }

    /*
        功能：在当前激光帧扫描起止时间范围内，计算某一时刻的旋转(相对于起始时刻的旋转增量)
    */
    void findRotation(double pointTime, float *rotXCur, float *rotYCur, float *rotZCur)
    {
        *rotXCur = 0; *rotYCur = 0; *rotZCur = 0;

        // 在imuTime下查找第一个大于当前点时刻pointTime的索引
        int imuPointerFront = 0;

        // 在imuDeskewInfo()中，对imuPointerCur进行计数(直到激光帧扫描结束的后0.01秒)
        // 遍历所有imuTime，按理说imuTime内的时间戳应该覆盖pointTime，以下只是保险起见，除非激光雷达坏了！！
        while (imuPointerFront < imuPointerCur)
        {
            if (pointTime < imuTime[imuPointerFront]) // 虽然不会发生，但是对时间要求真的严格！！
                break;
            ++imuPointerFront;
        }

        // 如果穷尽遍历imuTime，若imu时间没有覆盖这个点的时间，则异常赋值并推出
        if (pointTime > imuTime[imuPointerFront] || imuPointerFront == 0)
        {   
            // 异常退出之前，保留最接近当前点时刻pointTime的积分结果(imuDeskewInfo中)
            // 这样有什么用？？只能找个最接近pointTime的，但也不满足要求啊...
            *rotXCur = imuRotX[imuPointerFront];
            *rotYCur = imuRotY[imuPointerFront];
            *rotZCur = imuRotZ[imuPointerFront];
        } else {
            // 
            // 如果找到了时间戳最早大于pointTime的Imu积分内容，则通过插值得到当前时刻的旋转增量
            // imuPointerFront索引对应的时间戳最早大于pointTime，那么，imuPointerBack的时间戳是最晚小于pointTime的
            int imuPointerBack = imuPointerFront - 1;

            // 计算一下该点 --夹在两组Imu之间-- 的位置ratio
            double ratioFront = (pointTime - imuTime[imuPointerBack]) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            double ratioBack = (imuTime[imuPointerFront] - pointTime) / (imuTime[imuPointerFront] - imuTime[imuPointerBack]);
            
            // 前后百分比赋予旋转增量，这个增量是相对于 当前激光帧扫描开始时刻  第一个有效的imu增量里程计位姿 的
            *rotXCur = imuRotX[imuPointerFront] * ratioFront + imuRotX[imuPointerBack] * ratioBack;
            *rotYCur = imuRotY[imuPointerFront] * ratioFront + imuRotY[imuPointerBack] * ratioBack;
            *rotZCur = imuRotZ[imuPointerFront] * ratioFront + imuRotZ[imuPointerBack] * ratioBack;
        }
    }

    /*
        功能：平移插值
            如果传感器移动速度较慢，例如人行走的速度，那么可以认为激光在一帧时间范围内，平移量小到可以忽略不计
    */
    void findPosition(double relTime, float *posXCur, float *posYCur, float *posZCur)
    {
        *posXCur = 0; *posYCur = 0; *posZCur = 0;

        // If the sensor moves relatively slow, like walking speed, positional deskew seems to have little benefits. Thus code below is commented.

        // if (cloudInfo.odomAvailable == false || odomDeskewFlag == false)
        //     return;

        // float ratio = relTime / (timeScanEnd - timeScanCur);

        // *posXCur = ratio * odomIncreX;
        // *posYCur = ratio * odomIncreY;
        // *posZCur = ratio * odomIncreZ;
    }

    /*
        功能：激光点云运动畸变校正
        利用当前帧起止时刻之内的imu数据计算旋转增量，Imu里程计数据计算平移增量
        最终需要把所有点都投影到该激光帧第一个点所在的坐标系内！！
        relTime == laserCloudIn->points[i].time
    */
    PointType deskewPoint(PointType *point, double relTime)
    {
        // 这个来源于上文的时间戳通道和imu可用判断，没有或是不可用则返回点
        if (deskewFlag == -1 || cloudInfo.imuAvailable == false)
            return *point;

        // 点point的时间等于“该激光帧扫描初始时间 + laserCloudIn->points[i].time”
        double pointTime = timeScanCur + relTime;

        float rotXCur, rotYCur, rotZCur;
        findRotation(pointTime, &rotXCur, &rotYCur, &rotZCur);

        float posXCur, posYCur, posZCur;
        findPosition(relTime, &posXCur, &posYCur, &posZCur);

        // 检验这个点是否是该激光帧的第一个点
        // 每次调用cloudHandler都会启动resetParameters函数，通过析构实例对象重置firstPointFlag == true
        if (firstPointFlag == true)
        {
            // 第一个点的位姿增量(0)求逆，因为T(k)-W = (T(0)-W).inverse() * T(k)-(0) k在W系下
            transStartInverse = (pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur)).inverse();
            firstPointFlag = false;
        }

        // transform points to start
        // 将当前点Point变换到当前激光帧第一扫描点坐标系下，他这样“第一个点取逆再乘以后面点“的做法就很保险了，因为第一个扫描点和第一个有效的imu里程计位姿可能还不太一样(虽然差距很小)
        Eigen::Affine3f transFinal = pcl::getTransformation(posXCur, posYCur, posZCur, rotXCur, rotYCur, rotZCur);
        Eigen::Affine3f transBt = transStartInverse * transFinal;

        // 修正点云位置
        PointType newPoint;
        newPoint.x = transBt(0,0) * point->x + transBt(0,1) * point->y + transBt(0,2) * point->z + transBt(0,3);
        newPoint.y = transBt(1,0) * point->x + transBt(1,1) * point->y + transBt(1,2) * point->z + transBt(1,3);
        newPoint.z = transBt(2,0) * point->x + transBt(2,1) * point->y + transBt(2,2) * point->z + transBt(2,3);
        newPoint.intensity = point->intensity;

        return newPoint;
    }

    /*
        功能：将去畸变后的点云投影到range image，里面的坐标变换不太懂
    */
    void projectPointCloud()
    {
        int cloudSize = laserCloudIn->points.size();
        // range image projection
        for (int i = 0; i < cloudSize; ++i)
        {
            PointType thisPoint;
            // laserCloudIn就是原始的点云话题中的数据
            thisPoint.x = laserCloudIn->points[i].x;
            thisPoint.y = laserCloudIn->points[i].y;
            thisPoint.z = laserCloudIn->points[i].z;
            thisPoint.intensity = laserCloudIn->points[i].intensity;

            float range = pointDistance(thisPoint);
            if (range < lidarMinRange || range > lidarMaxRange)
                continue;

            int rowIdn = laserCloudIn->points[i].ring;
            // int rowIdn = (i % 64) + 1 ; // for MulRan dataset, Ouster OS1-64 .bin file,  giseop 

            if (rowIdn < 0 || rowIdn >= N_SCAN)
                continue;

            if (rowIdn % downsampleRate != 0)
                continue;

            float horizonAngle = atan2(thisPoint.x, thisPoint.y) * 180 / M_PI;
            
            // 水平角分辨率，Horizon_SCAN=1800，每格0.2度
            static float ang_res_x = 360.0/float(Horizon_SCAN);

            // 下面进行的一切操作就是想把扫描开始的地方角度0与角度360连在一起
            // horizonAngle 为[-180,180]，-90为[-270,90]，-round为[-90,270]，/ang_res_x为[-450,1350]，+Horizon_SCAN/2为[450,2250]
            // 经过以上变换，horizonAngle从[-180,180]映射到[450,2250]
            int columnIdn = -round((horizonAngle-90.0)/ang_res_x) + Horizon_SCAN/2;

            // 大于1800，则减去1800，相当于把1801～2250映射到1～450
            // 那么在数值上可以看出把columnIdn从horizonAngle:(-PI,PI]转换到columnIdn:[H/4,5H/4]
            // 然后判断columnIdn大小，把H到5H/4的部分切下来，补到0～H/4的部分
            // 相当于horizonAngle:(PI/2,PI]&&(-PI,PI/2]对应columnIdn：[0,H/4]&[H/4,H]
            // 这样就把扫描开始的地方角度为-90*与角度为90*的连在了一起，非常巧妙（）
            /*
                horizonAngle从[-180,180]映射到[450,2250]，大于1800，则减去1800，相当于把1801～2250映射到1～450

                假设前方是x，左侧是y，则正后方左边是180，右边是-180，以图像形式展开：(看不懂！！看懂了)
                                       0
                       90                        -90
                              180 || (-180)
                      (-180)   -----   (-90)  ------  0  ------ 90 -------180
                    变为:  90 ----180(-180) ---- (-90)  ----- (0)    ----- 90
            */
            if (columnIdn >= Horizon_SCAN)
                columnIdn -= Horizon_SCAN;

            if (columnIdn < 0 || columnIdn >= Horizon_SCAN)
                continue;

            if (rangeMat.at<float>(rowIdn, columnIdn) != FLT_MAX)
                continue;

            // 去畸变
            thisPoint = deskewPoint(&thisPoint, laserCloudIn->points[i].time);

            //图像中填入欧几里得深度
            // TODO： 不是在去畸变点云基础上投影的rangeimage，会不会存在误差
            rangeMat.at<float>(rowIdn, columnIdn) = range;

            // 转换成一维索引，存校正之后的激光点
            int index = columnIdn + rowIdn * Horizon_SCAN;
            fullCloud->points[index] = thisPoint;
        }
    }

    /*
        功能：根据range image的range提取有效点
        
        TODO 添加语义信息
        ************************************************
        ******  可以在这里使用RangeNet++筛选动态物体  ******
        ************************************************

        感觉需要修改cloud_info.msg，里边需要添加每一帧各个物体点云语义信息，在后面的特征提取和scan2map中根据语义进行匹配，甚至根据range image进行回环检测
        为完成这个工作，我需要删除lio-sam中的gps环节、阅读segMatch、阅读suma++、阅读F-loam(减少scan2map优化次数)

        如果说不用语义信息呢？简单的用一个DBSCAN聚类或普聚类到物体级，进行物体级配准的同时，用imu预测每个点的位置(检测是否在完全配准下，找到足够的近邻点(甚至在scan2map精确一下))，如果前后不一致，则删除动态物体
    */
    void cloudExtraction()
    {
        int count = 0; // 有效激光点数量
        // extract segmented cloud for lidar odometry
        for (int i = 0; i < N_SCAN; ++i)
        {
            // 为后面特征提取做准备
            // 提取特征的时候，每一行的前5个和最后5个不考虑(在有效点范围内)，但现在的range image仍然保留
            // 记录每根扫描线起始第5个激光点在一维数组中的索引，count在for外边，一直累计有效点数量
            cloudInfo.startRingIndex[i] = count - 1 + 5;

            for (int j = 0; j < Horizon_SCAN; ++j)
            {
                if (rangeMat.at<float>(i,j) != FLT_MAX)
                {
                    // mark the points' column index for marking occlusion later
                    // 记录激光点对应的Horizon_SCAN方向上的索引，为后续遮挡瑕点剔除做准备
                    cloudInfo.pointColInd[count] = j;

                    // save range info
                    // range存储
                    cloudInfo.pointRange[count] = rangeMat.at<float>(i,j);

                    // save extracted cloud
                    // 有效点存储
                    extractedCloud->push_back(fullCloud->points[j + i*Horizon_SCAN]);

                    // size of extracted cloud
                    ++count;
                }
            }
            // 记录每根扫描线倒数第5个激光点在一维数组中的索引
            cloudInfo.endRingIndex[i] = count -1 - 5;
        }
    }
    
    /*
        功能：发布去畸变且有效点的点云
    */
    void publishClouds()
    {
        cloudInfo.header = cloudHeader;
        // publishCloud在utility.h头文件中,需要传入发布句柄pubExtractedCloud
        // pubExtractedCloud定义在构造函数中，用来发布去畸变的点云
        // extractedCloud主要在cloudExtraction中被提取，点云被去除了畸变
        // cloudHeader.stamp 来源于currentCloudMsg,cloudHeader在cachePointCloud中被赋值currentCloudMsg.header
        // 在publishCloud函数中，返回sensor_msgs/PointCloud2类型，并发布extractedCloud，其中tempCloud.header.frame_id="base_link"(lidarFrame)
        cloudInfo.cloud_deskewed  = publishCloud(&pubExtractedCloud, extractedCloud, cloudHeader.stamp, lidarFrame);
        
        // 发布自定义cloud_info信息
        pubLaserCloudInfo.publish(cloudInfo);

        
        record_initialguesss(imu_initial_guess_pose); // fyx，不断刷新记录....还没有更好的解决方法
        // cout << "****************************************************" << endl;
        // cout << "Saving imu_guess_poses completely size as" << " " << imu_guess_rotation.size() << endl;
        
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lio_sam");

    ImageProjection IP;
    
    ROS_INFO("\033[1;32m----> Image Projection Started.\033[0m");


    // 对于一些只订阅一个话题的简单节点来说，我们使用ros::spin()进入接收循环，
    // 每当有订阅的话题发布时，进入回调函数接收和处理消息数据。
    // 但是更多的时候，一个节点往往要接收和处理不同来源的数据，并且这些数据的产生频率也各不相同，
    // 当我们在一个回调函数里耗费太多时间时，会导致其他回调函数被阻塞，导致数据丢失。
    // 这种场合需要给一个节点开辟多个线程，保证数据流的畅通。
    ros::MultiThreadedSpinner spinner(3); // 3线程
    spinner.spin(); // 始终等待消息回调
    
    return 0;
}
