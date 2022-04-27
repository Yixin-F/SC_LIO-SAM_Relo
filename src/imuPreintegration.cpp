#include "utility.h"

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/navigation/GPSFactor.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam_unstable/nonlinear/IncrementalFixedLagSmoother.h>

using gtsam::symbol_shorthand::X; // Pose3 (x,y,z,r,p,y)  // 这应该是因子图优化中的节点value
using gtsam::symbol_shorthand::V; // Vel   (xdot,ydot,zdot)
using gtsam::symbol_shorthand::B; // Bias  (ax,ay,az,gx,gy,gz)

/*
    订阅激光里程计(来自MapOptimization)和IMU里程计
    根据“前一时刻激光里程计”和“前一时刻到当前时刻的IMU里程计增量”计算“当前时刻的IMU里程计”
    rviz展示IMU里程计轨迹(局部)

    为什么进行预积分？
    利用Imu的加速度计和陀螺仪的数据，在减少计算量的情况下，对前后俩帧之间的位姿做一个估计
*/

/*
    功能：TransformFusion类实现多传感器tf融合，它继承了ParamServer类
        tf主要是负责不同Frame之间的变换，这里的一切变化都是为了rviz中的可视化
*/
class TransformFusion : public ParamServer
{
public:
    std::mutex mtx; // 最基本互斥锁，这里简单的定义了一把“锁”

    ros::Subscriber subImuOdometry;
    ros::Subscriber subLaserOdometry;

    ros::Publisher pubImuOdometry;
    ros::Publisher pubImuPath;

    /*
        Affine3f 仿射变换矩阵：实际上就是“平移向量+旋转变换”的组合，基本使用方法如下
            1）创建Eigen::Affine3f 对象a；
            2）创建类型为Eigen::Translation3f 对象b，用来存储平移向量；
            3）创建类型为Eigen::Quaternionf 四元数对象c，用来存储旋转变换；
            4）最后通过该方式生成最终Affine3f变换矩阵： a=b*c.toRotationMatrix();
            5）一个向量通过仿射变换时的方法是result_vector=test_affine*test_vector
        后面有一个odom2affine的函数，负责生成放射变换矩阵
    */
    Eigen::Affine3f lidarOdomAffine; 
    Eigen::Affine3f imuOdomAffineFront;
    Eigen::Affine3f imuOdomAffineBack;

    tf::TransformListener tfListener; // tf坐标变换监听
    tf::StampedTransform lidar2Baselink; // 暂存tf变换数据

    double lidarOdomTime = -1;

    /*
        nav_msgs/Odometry Message （里程计信息）：
            Header header
            string child_frame_id
            geometry_msgs/PoseWithCovariance pose
            geometry_msgs/TwistWithCovariance twist
    */
    deque<nav_msgs::Odometry> imuOdomQueue; // 可pop_front

    TransformFusion()
    {
        // 如果lidar系与base_link系不同(激光系和载体系)，需要外部提供两者之间的变换关系
        // 但是在yaml文件里两个坐标系的Frame是相同的，也就是说我们不需要监听此tf消息，但以防万一，有的bag确实是有tf消息类型的！！
        if(lidarFrame != baselinkFrame)
        {
            try
            {
                // 谁来发布lidar系与base_link系的tf消息？？
                tfListener.waitForTransform(lidarFrame, baselinkFrame, ros::Time(0), ros::Duration(3.0)); // 等待3s
                tfListener.lookupTransform(lidarFrame, baselinkFrame, ros::Time(0), lidar2Baselink); // 得到lidar系到base_link系的变换
            }
            catch (tf::TransformException ex)
            {
                ROS_ERROR("%s",ex.what());
            }
        }

        // 订阅激光历程计，来自mapOptimization
        subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry", 5, &TransformFusion::lidarOdometryHandler, this, ros::TransportHints().tcpNoDelay());
        
        // 订阅imu里里程计，来自IMUPreintegration.cpp中的类IMUPreintegration
        // 注意这边订阅的是odometry/imu_incremental，里面只有俩帧之间的imu预积分，一个增量内容
        // imuIntegratorImu_本身是个积分器，只有两帧之间的预积分，但是发布的时候发布的实际是结合了前述里程计本身有的位姿
        subImuOdometry   = nh.subscribe<nav_msgs::Odometry>(odomTopic+"_incremental",   2000, &TransformFusion::imuOdometryHandler,   this, ros::TransportHints().tcpNoDelay());
        
        // 发布imu里程计，用于rviz显示
        pubImuOdometry   = nh.advertise<nav_msgs::Odometry>(odomTopic, 2000);

        // 发布imu里程计轨迹？？
        pubImuPath       = nh.advertise<nav_msgs::Path>    ("lio_sam/imu/path", 1);
    }

    /*
        功能：订阅odom消息，并将其转化为变换矩阵
    */
    Eigen::Affine3f odom2affine(nav_msgs::Odometry odom)
    {
        double x, y, z, roll, pitch, yaw;
        x = odom.pose.pose.position.x; // nav_msgs::Odometry内部格式
        y = odom.pose.pose.position.y;
        z = odom.pose.pose.position.z;
        tf::Quaternion orientation;
        tf::quaternionMsgToTF(odom.pose.pose.orientation, orientation);
        tf::Matrix3x3(orientation).getRPY(roll, pitch, yaw);
        return pcl::getTransformation(x, y, z, roll, pitch, yaw);
    }

    /*
        功能：订阅lidarodom消息的回调函数，来自mapOptimization
        
        对std::lock_guard<std::mutex> lock(mtx)解释：
            对于最基本互斥锁mutex，需要lock()和unlock()进行上锁和解锁，很不方便
            而lock_guard导入mutex对象后会自动上锁，不规定mutex的生命周期，等lock_guard解析时mutex自动解锁
            在此，跳出以下两个函数作用域，lock_guard解析->mutex解锁
    */
    void lidarOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx); // 上锁，函数结束则解锁

        lidarOdomAffine = odom2affine(*odomMsg); // 最近时刻变换矩阵转换

        lidarOdomTime = odomMsg->header.stamp.toSec(); // 最近时刻时间戳
    }

    /*
        功能：订阅Imu里程计的回调函数，来自IMUPreintegration
            1）以--最近时刻激光里程计位姿--为基础，计算--该时刻--与当前时刻imu里程计计算位姿增量变换，相乘后得到当前时刻imu里程计位姿，提供imu位姿先验，并等待激光里程计的修正
            2）发布当前时刻imu里程计位姿，用于rviz展示；发布imu里程计路径(只是最近一帧激光里程计时刻与当前时刻之间的一段)
    */
    void imuOdometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        // static tf，每次进入函数时，不初始化
        // TransformBroadcaster用来发布tf消息，这里map系与odom系为同一个系(map_to_odom初始为0)
        static tf::TransformBroadcaster tfMap2Odom; // 后面没用到，感觉没必要，map系和odom系一般都是相同的
        static tf::Transform map_to_odom = tf::Transform(tf::createQuaternionFromRPY(0, 0, 0), tf::Vector3(0, 0, 0));
        // tf::Transform (input), stamp_ ( timestamp ), frame_id_ (frame_id), child_frame_id_(child_frame_id){ };
        tfMap2Odom.sendTransform(tf::StampedTransform(map_to_odom, odomMsg->header.stamp, mapFrame, odometryFrame));

        std::lock_guard<std::mutex> lock(mtx); // 上锁

        // 将最新imu里程计消息加入队列，由本cpp中的另一个类imuPreintegration来发布
        imuOdomQueue.push_back(*odomMsg); 

        // get latest odometry (at current IMU stamp)
        // 根据最近激光里程计时刻current时刻，从此刻其截取imu数据，用于计算imu增量位姿(此刻之前的imu数据已经经过激光里程计修正了)
        if (lidarOdomTime == -1)
            return;
        while (!imuOdomQueue.empty())
        {
            if (imuOdomQueue.front().header.stamp.toSec() <= lidarOdomTime)
                imuOdomQueue.pop_front(); // deque优于vector
            else
                break;
        }

        // 最近一帧激光里程计对应时刻的imu里程计位姿
        Eigen::Affine3f imuOdomAffineFront = odom2affine(imuOdomQueue.front());

        // 当前时刻imu里程计位姿，就是这一时间段imu增量末尾的位姿
        Eigen::Affine3f imuOdomAffineBack = odom2affine(imuOdomQueue.back());

        // imu增量位姿
        Eigen::Affine3f imuOdomAffineIncre = imuOdomAffineFront.inverse() * imuOdomAffineBack; 
        
        // 当前时刻imu里程计位姿(map坐标系下) = 最近的一帧激光里程计位姿 * imu里程计增量位姿变换 
        Eigen::Affine3f imuOdomAffineLast = lidarOdomAffine * imuOdomAffineIncre;  // 不用担心lidarOdomAffine被篡改，有互斥锁在！！

        // 得到xyz和rpy，有人说 pcl::getTranslationAndEulerAngles 存在精度问题，是因为这样导致的lio-sam在z方向漂移吗？感觉误差也不大
        float x, y, z, roll, pitch, yaw;
        pcl::getTranslationAndEulerAngles(imuOdomAffineLast, x, y, z, roll, pitch, yaw);
        
        // publish latest odometry 
        // 先监听了类IMUPreintegration中发布的odometry/imu_incremental，然后新发布发布名为"odometry/imu"的话题
        // 不再简单的发布imu增量，而是把x，y，z，roll，pitch，yaw都经过了激光里程计的修正(优化)，再发布
        nav_msgs::Odometry laserOdometry = imuOdomQueue.back();
        laserOdometry.pose.pose.position.x = x;
        laserOdometry.pose.pose.position.y = y;
        laserOdometry.pose.pose.position.z = z;
        laserOdometry.pose.pose.orientation = tf::createQuaternionMsgFromRollPitchYaw(roll, pitch, yaw);
        pubImuOdometry.publish(laserOdometry); // 这才是rviz里的imu先验里程计的显示

        // publish tf
        // 之前map和odom坐标系是固定重合的
        // 现在发布的tf消息是此时刻base_link相对于odom(map)系的关系，base_link始终是在随载体运动的，这边是为了rviz中tf_link的显示显示！！！
        // map优化提供激光，预积分提供imu，imu之间变换再乘以激光里程计得到各个时刻精确位姿
        static tf::TransformBroadcaster tfOdom2BaseLink;
        tf::Transform tCur;
        tf::poseMsgToTF(laserOdometry.pose.pose, tCur);
        if(lidarFrame != baselinkFrame)
            tCur = tCur * lidar2Baselink; // 借助lidar2Baselink，一直在监听
        tf::StampedTransform odom_2_baselink = tf::StampedTransform(tCur, odomMsg->header.stamp, odometryFrame, baselinkFrame);
        tfOdom2BaseLink.sendTransform(odom_2_baselink);

        // publish IMU path
        // 发布imu里程计路径(只是最近一帧激光里程计时刻与当前时刻之间的一段)
        static nav_msgs::Path imuPath;
        static double last_path_time = -1;
        double imuTime = imuOdomQueue.back().header.stamp.toSec();
        if (imuTime - last_path_time > 0.1) // 每隔0.1s加一个，imu的link系
        {
            last_path_time = imuTime;
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.stamp = imuOdomQueue.back().header.stamp;
            pose_stamped.header.frame_id = odometryFrame;
            pose_stamped.pose = laserOdometry.pose.pose;
            imuPath.poses.push_back(pose_stamped);

            // 删除最近一帧激光里程计时刻之前的imu里程计
            while(!imuPath.poses.empty() && imuPath.poses.front().header.stamp.toSec() < lidarOdomTime - 1.0)
                imuPath.poses.erase(imuPath.poses.begin());
            if (pubImuPath.getNumSubscribers() != 0)
            {
                imuPath.header.stamp = imuOdomQueue.back().header.stamp;
                imuPath.header.frame_id = odometryFrame;
                pubImuPath.publish(imuPath);
            }
        }
    }
};

/*
    功能：进行Imu预积分，提供imu里程计
        开始使用isam -> increasemental smoothing and mapping，加入IMU约束
*/
class IMUPreintegration : public ParamServer
{
public:

    std::mutex mtx; // 锁

    ros::Subscriber subImu;
    ros::Subscriber subOdometry;
    ros::Publisher pubImuOdometry;

    bool systemInitialized = false;

    // 各种噪声协方差模型，都声明成了对角矩阵，默认各维之间独立
    gtsam::noiseModel::Diagonal::shared_ptr priorPoseNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorVelNoise;
    gtsam::noiseModel::Diagonal::shared_ptr priorBiasNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise;
    gtsam::noiseModel::Diagonal::shared_ptr correctionNoise2;
    gtsam::Vector noiseModelBetweenBias;

    /**  imu预积分器(gtsam4.0后自带API)  **/
    // imuIntegratorOpt_负责预积分两个激光里程计之间的imu数据，作为约束加入因子图，并且优化bias
    gtsam::PreintegratedImuMeasurements *imuIntegratorOpt_;
    // imuIntegratorImu_用来根据最近激光里程计到达后优化好的bias，预测从该最近激光里程计到下一帧激光里程计之间的imu里程计增量
    gtsam::PreintegratedImuMeasurements *imuIntegratorImu_;

    /**  Imu数据队列  **/
    // 为imuIntegratorOpt_提供imu数据，比当前激光里程计数据早的imu一一预积分，然后逐个弹出 -> 预积分当前激光帧之前的imu，用于优化
    std::deque<sensor_msgs::Imu> imuQueOpt;
    // 为imuIntegratorImu_提供imu数据，从队头弹出当前激光里程计之前的imu数据，剩下的imu数据预积分用完一个弹一个 -> 预积分当前前激光帧之后的imu，在imuQueOpt优化基础上用来发布
    std::deque<sensor_msgs::Imu> imuQueImu;

    // imu因子图优化过程中的状态变量
    gtsam::Pose3 prevPose_; // A 3D pose (R,t) : (Rot3,Point3)
    gtsam::Vector3 prevVel_;
    gtsam::NavState prevState_; // Navigation state: Pose (rotation, translation) + velocity
    gtsam::imuBias::ConstantBias prevBias_; // All bias models live in the imuBias namespace

    // Imu状态
    gtsam::NavState prevStateOdom;
    gtsam::imuBias::ConstantBias prevBiasOdom;

    bool doneFirstOpt = false;
    double lastImuT_imu = -1;
    double lastImuT_opt = -1;

    // ISAM2优化器
    gtsam::ISAM2 optimizer; // 优化器
    gtsam::NonlinearFactorGraph graphFactors; // 总的因子图模型
    gtsam::Values graphValues; // 因子图模型中的值

    const double delta_t = 0;

    int key = 1; // 计数优化节点

    /**  imu-lidar位姿变换 **/
    // 注意：这里只包含了一个平移变换；同样的头文件的imuConverter也只有一个旋转变换
    // 事实上，作者是将imu数据先用imuConverter转换到“中间系”(与激光雷达系相差一个平移)；
    // 然后将激光雷达数据通过lidar2Imu平移到“中间系”，与imu数据对齐后；再imu2Lidar返回激光雷达系并发布
    // gtsam::Rot3(1, 0, 0, 0)是什么意思i，这个旋转是四元数形式吗？我只见过ypr形式，则会确实是四元数形式，单位寺四元数代表旋转，这里无旋转，则w=1；gtsam::Point3应该是平移
    gtsam::Pose3 imu2Lidar = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(-extTrans.x(), -extTrans.y(), -extTrans.z()));
    gtsam::Pose3 lidar2Imu = gtsam::Pose3(gtsam::Rot3(1, 0, 0, 0), gtsam::Point3(extTrans.x(), extTrans.y(), extTrans.z()));

    IMUPreintegration()
    {
        // 使用类和类内函数实现消息订阅，通用格式(4个参数)为：
        // sub = nh.subscribe<消息类型>(话题名称， 消息块缓存大小(一般等于发布消息队列长度)， 回调函数， this(回调函数是类内函数))

        // 订阅Imu原始数据，用下面因子图优化结果，施加俩帧之间的imu预积分量，预测每一时刻(imu频率)的imu里程计
        subImu      = nh.subscribe<sensor_msgs::Imu>  (imuTopic,                   2000, &IMUPreintegration::imuHandler,      this, ros::TransportHints().tcpNoDelay());

        // 订阅激光里程计，来自mapOptimization，用激光里程计两帧之间的imu预积分量构建因子图
        // 优化当前帧位姿，这个位姿仅用于更新每时每刻的imu里程计，以及下一次因子图优化
        subOdometry = nh.subscribe<nav_msgs::Odometry>("lio_sam/mapping/odometry_incremental", 5,    &IMUPreintegration::odometryHandler, this, ros::TransportHints().tcpNoDelay());

        // 发布Imu里程计：odometry/imu_incremental
        pubImuOdometry = nh.advertise<nav_msgs::Odometry> (odomTopic+"_incremental", 2000);

        // imu预积分的噪声协方差，boost::shared_ptr共享指针，与重力imuGravity有关？？是有关的！！
        boost::shared_ptr<gtsam::PreintegrationParams> p = gtsam::PreintegrationParams::MakeSharedU(imuGravity);
        // imuAccNoise和imuGyrNoise都是定义在头文件中的高斯白噪声，由配置文件中写入，定义为float形式
        p->accelerometerCovariance  = gtsam::Matrix33::Identity(3,3) * pow(imuAccNoise, 2); // acc white noise in continuous
        p->gyroscopeCovariance      = gtsam::Matrix33::Identity(3,3) * pow(imuGyrNoise, 2); // gyro white noise in continuous
        // 对于速度的积分误差？
        p->integrationCovariance    = gtsam::Matrix33::Identity(3,3) * pow(1e-4, 2); // error committed in integrating position from velocities
        // 假设没有初始bias，finished()意为设置完所有系数后返回构建的矩阵
        gtsam::imuBias::ConstantBias prior_imu_bias((gtsam::Vector(6) << 0, 0, 0, 0, 0, 0).finished());; // assume zero initial bias

        // 噪声先验，
        // Diagonal对角线矩阵，一般调用.finished(),注释中说finished()意为设置完所有系数后返回构建的矩阵
        priorPoseNoise  = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2).finished()); // rad,rad,rad,m, m, m
        priorVelNoise   = gtsam::noiseModel::Isotropic::Sigma(3, 1e4); // m/s，non-robust 各项同性噪声？？
        priorBiasNoise  = gtsam::noiseModel::Isotropic::Sigma(6, 1e-3); // 1e-2 ~ 1e-3 seems to be good
        
        // 激光里程计scan-to-map优化过程中发生退化，则选择一个较大的协方差
        correctionNoise = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 0.05, 0.05, 0.05, 0.1, 0.1, 0.1).finished()); // rad,rad,rad,m, m, m
        correctionNoise2 = gtsam::noiseModel::Diagonal::Sigmas((gtsam::Vector(6) << 1, 1, 1, 1, 1, 1).finished()); // rad,rad,rad,m, m, m
        noiseModelBetweenBias = (gtsam::Vector(6) << imuAccBiasN, imuAccBiasN, imuAccBiasN, imuGyrBiasN, imuGyrBiasN, imuGyrBiasN).finished(); // params in utility.h
        
        // imu预积分器，用于预测每一时刻（imu频率）的imu里程计（转到lidar系了，与激光里程计同一个系）
        imuIntegratorImu_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for IMU message thread
        // imu预积分器，用于因子图优化
        imuIntegratorOpt_ = new gtsam::PreintegratedImuMeasurements(p, prior_imu_bias); // setting up the IMU integration for optimization        
    }

    /*
        功能：初始化gtsam参数
    */
    void resetOptimization()
    {
        gtsam::ISAM2Params optParameters;
        optParameters.relinearizeThreshold = 0.1; // 差值大于0.1需要重新线性化
        optParameters.relinearizeSkip = 1; // 每当有1个值需要重新线性化时，对贝叶斯树进行更新
        optimizer = gtsam::ISAM2(optParameters);

        gtsam::NonlinearFactorGraph newGraphFactors; // 重新初始化因子图
        graphFactors = newGraphFactors;

        gtsam::Values NewGraphValues;
        graphValues = NewGraphValues;
    }

    void resetParams()
    {
        lastImuT_imu = -1;
        doneFirstOpt = false;
        systemInitialized = false;
    }

    /*
        功能：订阅激光里程计，“lio_sam/mapping/odometry_incremental”
        1）ConstPtr -> boost::shared_ptr<MSG const>，By passing a const pointer into the callback, we avoid doing a copy. 尤其是用于大量数据的回调
            再进行"常引用"，就不会改变“因共享指针而造成的原始数据变化”
        2）普通：Ptr -> boost::shared_ptr<MSG>

    */ 
    void odometryHandler(const nav_msgs::Odometry::ConstPtr& odomMsg)
    {
        std::lock_guard<std::mutex> lock(mtx); // 自动锁

        // 当前帧激光里程计时间戳
        double currentCorrectionTime = ROS_TIME(odomMsg);

        // make sure we have imu data to integrate
        // 确保imu优化队列中有imu数据进行预积分
        if (imuQueOpt.empty())
            return;

        // 当前帧激光里程计位姿，来自scan-to-map匹配、因子图优化后的位姿
        float p_x = odomMsg->pose.pose.position.x;
        float p_y = odomMsg->pose.pose.position.y;
        float p_z = odomMsg->pose.pose.position.z;
        float r_x = odomMsg->pose.pose.orientation.x;
        float r_y = odomMsg->pose.pose.orientation.y;
        float r_z = odomMsg->pose.pose.orientation.z;
        float r_w = odomMsg->pose.pose.orientation.w;

        // 判断是否在scam2map过程中退化，若退化则选择协方差较大的噪声
        // gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        bool degenerate = (int)odomMsg->pose.covariance[0] == 1 ? true : false; 

        // 只有转化成gtsam::Pose3形式，才能用于gtsam，同时“需要两个激光帧位姿才能计算一个激光scan2map的因子约束”
        gtsam::Pose3 lidarPose = gtsam::Pose3(gtsam::Rot3::Quaternion(r_w, r_x, r_y, r_z), gtsam::Point3(p_x, p_y, p_z));


        // 0. initialize system
        // 0. 系统初始化，第一帧激光帧
        if (systemInitialized == false)
        {
            // 重置ISAM2优化器
            resetOptimization();

            // pop old IMU message
            // 从imu优化队列中删除第一帧激光里程计时刻之间的Imu数据，delta_t=0
            while (!imuQueOpt.empty())
            {
                if (ROS_TIME(&imuQueOpt.front()) < currentCorrectionTime - delta_t)
                {
                    lastImuT_opt = ROS_TIME(&imuQueOpt.front());
                    imuQueOpt.pop_front();
                }
                else
                    break;
            }

            // initial pose
            // 添加里程计位姿先验因子，--含rot和trans--
            // lidarPose为本回调函数受到的scan2map优化后的激光里程计数据，重组成gtsam的pose格式(前面)
            // 将lidarPose通过lidar2Imu(只含平移)转换到“中间坐标系”，gtsam中compose应该是类似于左乘的含义
            prevPose_ = lidarPose.compose(lidar2Imu); 
            // 在初始化的先验因子时，pose对应X，速度对应V，bias对应B，这边仅定义了一个位姿先验priorPose，它是PriorFactor的一种
            // 这里用的先验位姿是取自mapOptimization中的优化激光里程计，合情合理
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, priorPoseNoise);
            // 将位姿先验priorPose通过add加入总的因子图graphFactors，理所当然
            graphFactors.add(priorPose);

            // initial velocity
            // 添加里程计速度先验因子
            prevVel_ = gtsam::Vector3(0, 0, 0);
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, priorVelNoise);
            graphFactors.add(priorVel);

            // initial bias
            // 添加imu偏置(零偏)
            prevBias_ = gtsam::imuBias::ConstantBias();
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, priorBiasNoise);
            graphFactors.add(priorBias);

            // add values
            // 变量节点赋值
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_); // imu偏置也作为优化节点，我们也要进行Imu偏置优化的，这就是所谓的“imu与lidar的禁耦合”！！

            // optimize once
            // 优化一次
            optimizer.update(graphFactors, graphValues);
            // 因子和节点均清零，为什么要清零不能继续用吗？
            // 是因为节点信息保存在gtsam::ISAM2 optimizer，所以要清零后才能继续使用
            // 每一阶段的约束信息只需优化一次，所以每次优化结束后，要清空isam中的所有约束信息
            graphFactors.resize(0);
            graphValues.clear();

            // 积分器重置，重置优化后的imu偏置
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBias_);
            imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);
            
            key = 1; // 第一帧加入了因子图
            systemInitialized = true; // 初始化完成
            return;
        }


        // reset graph for speed
        // 每隔100帧激光里程计，重置ISAM2优化器，保证优化效率
        // TODO： 这样直接丢掉前100帧会不会太鲁莽？？
        if (key == 100)
        {
            // get updated noise before reset
            // 在重置前，保留最近一帧激光里程计的位姿、速度、偏置噪声模型，像是边缘化协方差
            gtsam::noiseModel::Gaussian::shared_ptr updatedPoseNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(X(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedVelNoise  = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(V(key-1)));
            gtsam::noiseModel::Gaussian::shared_ptr updatedBiasNoise = gtsam::noiseModel::Gaussian::Covariance(optimizer.marginalCovariance(B(key-1)));
            
            // reset graph
            // 重置ISAM2优化器
            resetOptimization();

            // add pose
            // 添加位姿先验因子，用最近一帧的值作为先验
            // 重置之后的初始化与第一帧的初始化步骤基本相同，但是先验值都是继承于最近一帧的
            gtsam::PriorFactor<gtsam::Pose3> priorPose(X(0), prevPose_, updatedPoseNoise);
            graphFactors.add(priorPose);
            // add velocity
            gtsam::PriorFactor<gtsam::Vector3> priorVel(V(0), prevVel_, updatedVelNoise);
            graphFactors.add(priorVel);
            // add bias
            gtsam::PriorFactor<gtsam::imuBias::ConstantBias> priorBias(B(0), prevBias_, updatedBiasNoise);
            graphFactors.add(priorBias);
            // add values
            graphValues.insert(X(0), prevPose_);
            graphValues.insert(V(0), prevVel_);
            graphValues.insert(B(0), prevBias_);
            // optimize once
            optimizer.update(graphFactors, graphValues);
            graphFactors.resize(0);
            graphValues.clear();

            key = 1; // 重置为1
        }


        // 1. integrate imu data and optimize
        // 1. --逐个--计算前一激光帧与当前激光帧之间的imu预积分量，用前一帧的状态施加预积分量得到当前帧初始状态估计
        // 添加来自mapOptimization的当前帧位姿，进行因子图优化，更新当前帧状态
        while (!imuQueOpt.empty()) // 被激光里程计优化后的imu队列不为空
        {
            // pop and integrate imu data that is between two optimizations
            // 提取前一帧与当前帧之间的Imu数据，计算预积分
            sensor_msgs::Imu *thisImu = &imuQueOpt.front();
            double imuTime = ROS_TIME(thisImu);
            // currentCorrectionTime是当前回调函数收到的激光里程计数据的时间
            if (imuTime < currentCorrectionTime - delta_t)
            {
                double dt = (lastImuT_opt < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_opt);
                // imu预积分数据输入：加速度、角速度、dt
                // 加入的是这个用来因子图优化的预积分器imuIntegratorOpt_，注意加入了上一步算出的dt
                // 作者要求的9轴imu数据中的欧拉角在imuPreintegration.cpp没有用到，“3轴的位姿”全在地图优化里用到的
                imuIntegratorOpt_->integrateMeasurement(
                        gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                
                // 在推出一次数据前保存上一个数据的时间戳
                lastImuT_opt = imuTime;

                // 从队列中删除已经处理的imu数据
                imuQueOpt.pop_front();
            }
            else
                break;
        }

        // add imu factor(pose&velocity) to graph
        // 利用两激光帧之间的所有imu数据完成了预积分后，增加imu因子到因子图中
        // 注意后面容易被遮挡，imuIntegratorOpt_的值经过格式转换被传入preint_imu中
        // 因此可以推测imuIntegratorOpt_中的integrateMeasurement函数应该就是一个简单的积分轮子，传入数据和dt，得到一个积分量,数据会被存放在imuIntegratorOpt_中
        const gtsam::PreintegratedImuMeasurements& preint_imu = dynamic_cast<const gtsam::PreintegratedImuMeasurements&>(*imuIntegratorOpt_);
        
        // imuFactor参数：前一帧位姿，前一帧速度，当前帧位姿，当前帧速度，前一帧偏置，预计分量
        gtsam::ImuFactor imu_factor(X(key - 1), V(key - 1), X(key), V(key), B(key - 1), preint_imu);
        graphFactors.add(imu_factor);

        // add imu bias between factor
        // 添加imu偏置因子，前一帧偏置B(key - 1)，当前帧偏置B(key)，观测值，噪声协方差；deltaTij()是积分段的时间
        graphFactors.add(gtsam::BetweenFactor<gtsam::imuBias::ConstantBias>(B(key - 1), B(key), gtsam::imuBias::ConstantBias(),
                         gtsam::noiseModel::Diagonal::Sigmas(sqrt(imuIntegratorOpt_->deltaTij()) * noiseModelBetweenBias)));
        
        // add lidar pose factor
        // 添加激光位姿因子
        gtsam::Pose3 curPose = lidarPose.compose(lidar2Imu);
        gtsam::PriorFactor<gtsam::Pose3> pose_factor(X(key), curPose, degenerate ? correctionNoise2 : correctionNoise);
        graphFactors.add(pose_factor);
        
        // insert predicted values
        // 用前一帧的状态、偏置，得到当前帧的状态
        // Navigation state: Pose (rotation, translation) + velocity
        gtsam::NavState propState_ = imuIntegratorOpt_->predict(prevState_, prevBias_);
        
        // 变量节点赋初值
        graphValues.insert(X(key), propState_.pose()); // (rotation, translation)
        graphValues.insert(V(key), propState_.v()); // velocity
        graphValues.insert(B(key), prevBias_); // prevBias_

        // optimize
        optimizer.update(graphFactors, graphValues);
        optimizer.update();
        graphFactors.resize(0);
        graphValues.clear();

        // Overwrite the beginning of the preintegration for the next step.
        // 得到最后的优化结果
        gtsam::Values result = optimizer.calculateEstimate();
        // 更新当前位姿、速度，为下一次k=100时重置做准备，提供先验
        prevPose_  = result.at<gtsam::Pose3>(X(key));
        prevVel_   = result.at<gtsam::Vector3>(V(key));
        // 更新当前帧状态
        prevState_ = gtsam::NavState(prevPose_, prevVel_);
        // 更新当前帧imu偏置
        prevBias_  = result.at<gtsam::imuBias::ConstantBias>(B(key));

        // Reset the optimization preintegration object.
        // 重置积分器，设置新的偏置，这样下一帧激光里程计进来的时候，预积分量就是两帧之间的增量
        imuIntegratorOpt_->resetIntegrationAndSetBias(prevBias_);

        // check optimization
        // imu因子图优化结果，速度或偏置过大，认为失败
        if (failureDetection(prevVel_, prevBias_))
        {
            resetParams();
            return;
        }


        // 2. after optiization, re-propagate imu odometry preintegration
        // 2. 优化过后，进行重传播；优化更新了imu的偏置
        // 用新的偏置重新计算当前激光里程计时刻之后的imu预积分，这个预积分用于计算每时刻位姿
        prevStateOdom = prevState_;
        prevBiasOdom  = prevBias_;

        // first pop imu message older than current correction data
        // 首先，从imu队列中删除当前激光里程计时刻之间的imu数据
        double lastImuQT = -1;
        // 注意，这里imuQueImu是要“删除”当前激光帧“之前”的imu数据，是想根据当前帧“之后”的累计递推。
        // 而前面的imuIntegratorOpt_做的事情是，“提取”当前帧“之前”的imu数据，用两帧激光里程计之间的imu数据进行积分，处理过后就弹出
        // 因此每来一帧新的激光里程计数据时，imuQueOpt队列变化为：“新的激光里程计帧--之前--的imu数据“提取出来，做一次预积分，用一个删一个
        // 而imuQueImu的操作是当“新的一帧激光里程计来到时，把之前的imu数据直接删掉，仅保留当前帧--之后--的imu数据“，用作下一次新激光里程计未来到时间里imu增量式里程计的预测
        // imuQueImu和imuQueOpt的区别要明确,imuIntegratorImu_和imuIntegratorOpt_的区别也要明确,实际上imuIntegratorImu_在用imuIntegratorOpt_的优化结果，见imuhandler中的注释
        while (!imuQueImu.empty() && ROS_TIME(&imuQueImu.front()) < currentCorrectionTime - delta_t)
        {
            lastImuQT = ROS_TIME(&imuQueImu.front());
            imuQueImu.pop_front();
        }

        // repropogate
        // imuIntegratorImu_再一次对剩余的imu数据计算预积分，有什么用途呢？后面回调imuHandler()中对每一个Imu数据也进行了预积分，这次muIntegratorImu_传播是做什么的？
        // 这是只对当前接收到的imu数据进行预积分！！！上面imuIntegratorImu_还有一次积分，是对当前帧之后剩余的imu全都预积分，两者并不冲突，前后连接很完美
        if (!imuQueImu.empty())
        {
            // reset bias use the newly optimized bias
            // 传入状态，重置预积分器和最新的偏置
            imuIntegratorImu_->resetIntegrationAndSetBias(prevBiasOdom);

            // integrate imu message from the beginning of this optimization
            // 根据这一次优化更新，继续计算后面的预积分
            // 利用imuQueImu中的数据进行预积分，之前imuQueOpt的通过imuIntegratorOpt_更新了bias
            for (int i = 0; i < (int)imuQueImu.size(); ++i)
            {
                sensor_msgs::Imu *thisImu = &imuQueImu[i];
                double imuTime = ROS_TIME(thisImu);
                double dt = (lastImuQT < 0) ? (1.0 / 500.0) :(imuTime - lastImuQT);

                // 注意：加入的是这个用于传播的预积分器imuIntegratorImu_(之前用来计算的是imuIntegratorOpt_)
                // 注意加入了上一步算出的dt，结果被存放在imuIntegratorImu_中
                imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu->linear_acceleration.x, thisImu->linear_acceleration.y, thisImu->linear_acceleration.z),
                                                        gtsam::Vector3(thisImu->angular_velocity.x,    thisImu->angular_velocity.y,    thisImu->angular_velocity.z), dt);
                lastImuQT = imuTime;
            }
        }

        ++key;

        //设置成True，用来通知另一个负责发布imu里程计的回调函数imuHandler“可以发布了”
        doneFirstOpt = true;
    }

    /*
        功能：若imu数据速度或者偏置过大，则认为imu因子图优化失败，需要进行重置
        TODO 这或许是防抖关键，添加个地面优化来进行约束平滑？？
    */
    bool failureDetection(const gtsam::Vector3& velCur, const gtsam::imuBias::ConstantBias& biasCur)
    {
        Eigen::Vector3f vel(velCur.x(), velCur.y(), velCur.z());
        if (vel.norm() > 30)
        {
            ROS_WARN("Large velocity, reset IMU-preintegration!");
            return true;
        }

        Eigen::Vector3f ba(biasCur.accelerometer().x(), biasCur.accelerometer().y(), biasCur.accelerometer().z());
        Eigen::Vector3f bg(biasCur.gyroscope().x(), biasCur.gyroscope().y(), biasCur.gyroscope().z());
        if (ba.norm() > 1.0 || bg.norm() > 1.0)
        {
            ROS_WARN("Large bias, reset IMU-preintegration!");
            return true;
        }

        return false;
    }

    /*
        功能：订阅imu原始数据
             1）用上一帧激光里程计时刻对应的状态、偏置，施加从该时刻的imu预积分量，得到当前时刻的状态，也就是imu里程计
             2）imu里程计位姿转到激光雷达系，发布imu增量里程计
    */
    void imuHandler(const sensor_msgs::Imu::ConstPtr& imu_raw)
    {
        std::lock_guard<std::mutex> lock(mtx); // 加锁

        // imu原始数据转换到lidar系(其实是“中间系”)，包括加速度、角速度、rpy
        sensor_msgs::Imu thisImu = imuConverter(*imu_raw);

        // 添加Imu数据到队列
        // 两个双端队列一开始存储的imu是完全相同的，只是后来的用途不同；imuQueOpt是预积分加入imu位姿约束因子的，imuQueImu是在imuQueOpt优化位姿的基础上预积分得到imu先验位姿
        imuQueOpt.push_back(thisImu);
        imuQueImu.push_back(thisImu);

        // 要求上一次imu因子图优化执行成功，确保上一帧
        if (doneFirstOpt == false)
            return;

        double imuTime = ROS_TIME(&thisImu);
        double dt = (lastImuT_imu < 0) ? (1.0 / 500.0) : (imuTime - lastImuT_imu);
        lastImuT_imu = imuTime;

        // integrate this single imu message
        imuIntegratorImu_->integrateMeasurement(gtsam::Vector3(thisImu.linear_acceleration.x, thisImu.linear_acceleration.y, thisImu.linear_acceleration.z),
                                                gtsam::Vector3(thisImu.angular_velocity.x,    thisImu.angular_velocity.y,    thisImu.angular_velocity.z), dt);

        // predict odometry
        gtsam::NavState currentState = imuIntegratorImu_->predict(prevStateOdom, prevBiasOdom);

        // publish odometry
        nav_msgs::Odometry odometry;
        odometry.header.stamp = thisImu.header.stamp;
        odometry.header.frame_id = odometryFrame;
        odometry.child_frame_id = "odom_imu";

        // transform imu pose to ldiar
        // 发布的imu里程计与lidar系对齐
        gtsam::Pose3 imuPose = gtsam::Pose3(currentState.quaternion(), currentState.position());
        gtsam::Pose3 lidarPose = imuPose.compose(imu2Lidar);

        odometry.pose.pose.position.x = lidarPose.translation().x();
        odometry.pose.pose.position.y = lidarPose.translation().y();
        odometry.pose.pose.position.z = lidarPose.translation().z();
        odometry.pose.pose.orientation.x = lidarPose.rotation().toQuaternion().x();
        odometry.pose.pose.orientation.y = lidarPose.rotation().toQuaternion().y();
        odometry.pose.pose.orientation.z = lidarPose.rotation().toQuaternion().z();
        odometry.pose.pose.orientation.w = lidarPose.rotation().toQuaternion().w();
        
        odometry.twist.twist.linear.x = currentState.velocity().x();
        odometry.twist.twist.linear.y = currentState.velocity().y();
        odometry.twist.twist.linear.z = currentState.velocity().z();
        odometry.twist.twist.angular.x = thisImu.angular_velocity.x + prevBiasOdom.gyroscope().x();
        odometry.twist.twist.angular.y = thisImu.angular_velocity.y + prevBiasOdom.gyroscope().y();
        odometry.twist.twist.angular.z = thisImu.angular_velocity.z + prevBiasOdom.gyroscope().z();
        pubImuOdometry.publish(odometry);
    }
};


int main(int argc, char** argv)
{
    ros::init(argc, argv, "roboat_loam");

    // 类创建对象时，各种回调函数就会生效
    IMUPreintegration ImuP;

    // tf消息主要用于点云frame_id像不同坐标系下的转换，比如从base_link转换到世界坐标系
    // 同时，在tf/message_filter中可以暂存tf消息，直到有相同时间戳的数据来到时，进行多种数据的同步(lio-sam未使用)
    TransformFusion TF;

    ROS_INFO("\033[1;32m----> IMU Preintegration Started.\033[0m");
    
    // 防阻塞多线程回调函数
    // 该节点订阅了多个消息，如果某个话题回调时间过长，会导致其他话题回调的阻塞，所以需要开辟多个线程，保证数据流畅
    // 使用.spin() 是指不返回(始终等待回调函数执行)，直到ros关闭(Ctrl+c)；若用.spinOnce() 则需要设置调用频率(书写ros::ok()循环)，同时注意消息块大小与发布频率，防止数据丢失与延迟
    ros::MultiThreadedSpinner spinner(4);
    spinner.spin();
    
    return 0;
}
