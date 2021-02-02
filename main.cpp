// Copyright 2020 Mykhailo Bondarenko
#include <iostream>
#include <sys/stat.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#define DATA_TYPE float  // you can place bool here

// using namespace cv;
using cv::Mat;
using cv::Rect;
using cv::imread;
using cv::String;
using cv::waitKey;
using cv::Point2f;
using cv::moveWindow;
using cv::warpAffine;
using cv::namedWindow;
using cv::resizeWindow;
using cv::getTickCount;
using cv::WINDOW_NORMAL;
using cv::IMREAD_GRAYSCALE;
using cv::getTickFrequency;
using cv::destroyAllWindows;
using cv::getRotationMatrix2D;
// using namespace std;
using std::max;
using std::min;
using std::cin;
using std::cout;
using std::pair;
using std::string;
using std::vector;
using std::is_same;
using std::make_pair;
using std::stringstream;

// ===================constants

const int WINDOW_SIZE = 140;
const int LOG_DELAY = 500;
const char* IMAGE_LOCATION_PREFIX = "";

// ===================declarations

class BinaryColon{
 public:
    int features_num;
    int clusters_num;
    vector<DATA_TYPE*> data;
    vector<float*> cluster_points;
    int* cluster_sizes;
    int capacity;
    int points_num;
    int cluster_targ_size = 80;

    BinaryColon(int fn, int cn);

    void feed(float* point);

    float* get_cluster_distances(float* point);
};

template <typename T>  // T is either vector<bool*> or vector<float*>
vector<vector<T> > get_clusters(
    vector<T> data, vector<float*> cluster_points,
        int clusters_num, int features_num);

// ===================printing functions and delay function

void print_progress(int i, int iter, int time_began) {
    int sec = ((double) iter - i) * (
        ((double) time(0) - time_began) / ((double) max(i, 1)));
    cout << "\r" << (int) ((double) i / ((double)iter / 100)) << "% "
                 << sec / 3600 << "h:" 
                 << (sec - (sec / 3600) * 3600) / 60 << "m:"
                 << (sec - (sec / 60) * 60) << "s            ";
    cout.flush();
}

template<typename T>
std::string to_string(T val) {
    stringstream ss;
    ss << val;
    return ss.str();
}

bool log_delay(uint64 *prev_log_time) {
    double ms = ((double) getTickCount()) / getTickFrequency() * 1000.0;
    if((*prev_log_time) != ((int64)ms / LOG_DELAY) * LOG_DELAY) {
        (*prev_log_time) = ((int64)ms / LOG_DELAY) * LOG_DELAY;
        return true;
    } 
    return false;
}

void print_clusters(BinaryColon &bc) {
    vector<vector<DATA_TYPE*>> clusters = get_clusters(
        bc.data, bc.cluster_points, bc.clusters_num, bc.features_num);
    for (int cli = 0; cli < bc.clusters_num; cli++) {
        cout << clusters[cli].size() << "\n";
        for (DATA_TYPE* p : clusters[cli]) {
            for (int c = 0; c < bc.features_num; c++) {
                cout << p[c] << " ";
            }
            cout << "\n";
        }
        cout << "\n";
    }
    clusters.clear();
}

// ===================showing functions

template <typename T> //  T is either bool* of float*
void feat_show(T point, int res, String winname) {
    if (is_same<T, float*>::value) {
        Mat im(res, res, CV_32FC1, point);
        im.convertTo(im, CV_8U, 255.0);
        imshow(winname, im);
    } else {
        if (is_same<T, bool*>::value) {
            Mat im(res, res, CV_8UC1, point);
            im.convertTo(im, CV_8U, 255.0);
            imshow(winname, im);
        }
    }
}

void create_cluster_windows(int clusters_num) {
    int x = -50; 
    int y = WINDOW_SIZE + 30;
    for (int i = 0; i < clusters_num; i++) {
        x += WINDOW_SIZE;
        if (x > 1300) {
            x = -50 + WINDOW_SIZE;
            y += WINDOW_SIZE + 20;
        }
        namedWindow("cluster_"+to_string(i), WINDOW_NORMAL);
        moveWindow("cluster_"+to_string(i), x, y);
    }
}

void show_clusters(BinaryColon &bc) {
    for (int i = 0; i < bc.clusters_num; i++) {
        feat_show(bc.cluster_points[i], (int)sqrt((float)bc.features_num), "cluster_"+to_string(i));
        resizeWindow("cluster_"+to_string(i), WINDOW_SIZE, WINDOW_SIZE);
    }
    waitKey(1);
}

// ===================Mat working functiions

float * Mat_to_float(Mat& I) {
    float * point = new float[I.rows * I.cols];
    float * p;
    for (int i = 0; i < I.rows; i++) {
        p = I.ptr<float>(i);
        for (int j = 0; j < I.cols; j++) {
            point[i * I.cols + j] = p[j];
        }
    }
    return point;
}

void fill_Mat_with_float(Mat& I, float* fill_with) {
    float * p;
    for (int i = 0; i < I.rows; i++) {
        p = I.ptr<float>(i);
        for (int j = 0; j < I.cols; j++) {
            p[j] = fill_with[i * I.cols + j];
        }
    }
}

void Mat_rotate(Mat &src, Mat &dst, int angle) {
    Point2f pc(src.cols/2., src.rows/2.);
    Mat r = getRotationMatrix2D(pc, angle, 1.0);
    warpAffine(src, dst, r, src.size());
}

void Mat_brightness(Mat &I, float brightness) {
    float * p;
    float cur_brightness = 0; 
    float max_br = 0;
    float min_br = 1;
    for (int i = 0; i < I.rows; i++) {
        p = I.ptr<float>(i);
        for (int j = 0; j < I.cols; j++) {
            cur_brightness += p[j];
            max_br = max(max_br, p[j]);
            min_br = min(min_br, p[j]);
        }
    }
    cur_brightness /= I.rows * I.cols;
    float distort = max(cur_brightness - min_br, max_br - cur_brightness);
    float distort_new = min(brightness - 0, 1 - brightness);
    float distort_k = distort_new / distort;
    for (int i = 0; i < I.rows; i++) {
        p = I.ptr<float>(i);
        for (int j = 0; j < I.cols; j++) {
            p[j] -= cur_brightness;
            p[j] *= distort_k;
            p[j] += brightness;
        }
    }
}

void Mat_combine(Mat &I, Mat *W, int n) {
    float * p;
    // int min_rows = W[0].rows;
    // int min_cols = W[0].cols;
    for (int i = 0; i < W[n - 1].rows; i++) {
        p = I.ptr<float>(i);
        for (int j = 0; j < W[n - 1].cols; j++) {
            p[j] = 0;
            for (int Wi = 0; Wi < n; Wi ++) {
                p[j] += W[Wi].at<float>(i, j);
            }
            p[j] /= n;
        }
    }
}

// ===================BinaryColon IO functions

void save_bc(BinaryColon &bc, string file) {
    FILE * f = fopen(file.c_str(), "w");

    fputs((to_string(bc.features_num) + "\n").c_str(), f);
    fputs((to_string(bc.clusters_num) + "\n\n").c_str(), f);

    fputs((to_string(bc.data.size()) + "\n").c_str(), f);
    for (DATA_TYPE* p : bc.data) {
        for (int c = 0; c < bc.features_num; c++)
            fputs((to_string(p[c]) + " ").c_str(), f);
        fputs("\n", f);
    }
    fputs("\n", f);

    fputs((to_string(bc.cluster_points.size()) + "\n").c_str(), f);
    for (float* p : bc.cluster_points) {
        for (int c = 0; c < bc.features_num; c++)
            fputs((to_string(p[c]) + " ").c_str(), f);
        fputs("\n", f);
    }
    fputs("\n", f);

    for (int ci = 0; ci < bc.clusters_num; ci++) {
        fputs((to_string(bc.cluster_sizes[ci]) + " ").c_str(), f);
    }
    fputs("\n\n", f);

    fputs((to_string(bc.capacity) + "\n").c_str(), f);
    fputs((to_string(bc.points_num) + "\n").c_str(), f);
    fputs((to_string(bc.cluster_targ_size) + "\n").c_str(), f);

    fclose(f);
}

BinaryColon read_bc(string file) {
    FILE * f = fopen(file.c_str(), "r");
    int len = 0;
    DATA_TYPE * point_b;
    float * point_f;
    BinaryColon bc(0, 0);

    fscanf(f, "%i", &bc.features_num);
    fscanf(f, "%i", &bc.clusters_num);

    fscanf(f, "%i", &len);
    for (int pi = 0; pi < len; pi++) {
        point_b = new DATA_TYPE[bc.features_num];
        for (int c = 0; c < bc.features_num; c++) {
            float i; fscanf(f, "%f", &i);
            point_b[c] = static_cast<DATA_TYPE>(i);
        }
        bc.data.push_back(point_b);
    }

    fscanf(f, "%i", &len);
    for (int pi = 0; pi < len; pi++) {
        point_f = new float[bc.features_num];
        for (int c = 0; c < bc.features_num; c++) {
            fscanf(f, "%f", &point_f[c]);
        }
        bc.cluster_points.push_back(point_f);
    }

    bc.cluster_sizes = new int[bc.clusters_num];
    for (int ci = 0; ci < bc.clusters_num; ci++)
        fscanf(f, "%i", &bc.cluster_sizes[ci]);

    fscanf(f, "%i", &bc.capacity);
    fscanf(f, "%i", &bc.points_num);
    fscanf(f, "%i", &bc.cluster_targ_size);

    fclose(f);

    return bc;
}

bool file_exists(string file) {
    struct stat buffer;
    return (stat (file.c_str(), &buffer) == 0); 
}

// ===================math functions

float rand01() {
    return (float) rand() / RAND_MAX;
}

bool prob(float p) {
    return rand01() < p;
}

template <typename T>  // T is either bool* of float*
float arr_sum(T begin, T end) {
    double s = 0;
    for (; begin != end; begin++) {
        s += *begin;
    }
    return (float) s;
}

template <typename Tp1, typename Tp2>  // Tp1 & Tp2 are either bool* of float*
float dej_dist(Tp1 p1, Tp2 p2, int features_num) {
    /// returns square of dejkstra distance from p1 to p2. Ranged to be in 0..1
    double dist = 0;
    for (int i = 0; i != features_num; i++) {
        dist += pow(static_cast<double>(p1[i]) - static_cast<double>(p2[i]), 2);
    }
    return static_cast<float>(dist / static_cast<double>(features_num));
}

DATA_TYPE* float_to_binary(float* fl, int n) {
    DATA_TYPE* binary = new DATA_TYPE[n];
    float point_mean = (float) arr_sum(fl, fl + n) / n;
    for (int i = 0; i!=n; i++) {
        binary[i] = fl[i] > point_mean;
    }
    return binary;
}

template<typename T>  // T is either float or double, elements in range 0..1
T* reduce_dimentionality(T* arr, int n, int dim) {
    T* arr_red = new T[n];
    dim -= 1;
    T dim_T = static_cast<T>(dim);
    for(int i = 0; i < n; i++)
        arr_red[i] = static_cast<T>(round(static_cast<T>(arr[i] * dim)) / dim_T);
    return arr_red;
}

template<typename Tarr, typename Tmult, typename Tadd>
void arr_multiply_and_add(Tarr arr_b, Tarr arr_e, Tmult m, Tadd a) {
    for (; arr_b != arr_e; arr_b ++)
        (*arr_b) = (*arr_b) * m + a;
}

// ===================k-means functions

template <typename T>  // T is either vector<bool*> or vector<float*>
float* mean_point(T points, int features_num) {
    int points_l = points.size();
    if (points_l > 0) {
        double* point = new double[features_num];
        for (int i = 0; i != features_num; i++)
            point[i] = 0;
        for (auto p : points) {
            for (int i = 0; i != features_num; i++)
                point[i] += static_cast<double>(
                    p[i]) / static_cast<double>(points_l);
        }
        float* point_fl = new float[features_num];
        for (int i = 0; i != features_num; i++)
            point_fl[i] = static_cast<float>(point[i]);
        delete [] point;
        return point_fl;
    } else {
        float* point = new float[features_num];
        for (int i = 0; i != features_num; i++)
            point[i] = rand01();
        return point;
    }
}

template <typename T>  // T is either vector<bool*> or vector<float*>
float* mean_point_speedmod(
        T points, float* cluster_point, int features_num, float k) {
    // this function allows to regulate the learning rate
    int points_l = points.size();
    if (points_l > 0) {
        double* point = new double[features_num];
        for (int i = 0; i != features_num; i++)
            point[i] = 0;
        for (auto p : points) {
            for (int i = 0; i != features_num; i++)
                point[i] += static_cast<double>(
                    p[i]) / static_cast<double>(points_l);
        }
        float* point_fl = new float[features_num];
        for (int i = 0; i != features_num; i++)
            point_fl[i] = static_cast<float>(
                (point[i] * k) + (cluster_point[i] * (1 - k)));
        delete [] point;
        return point_fl;
    } else {
        float* point = new float[features_num];
        for (int i = 0; i != features_num; i++)
            point[i] = rand01();
        return point;
    }
}

template <typename T>  // T is either bool* or float*
vector<vector<T> > get_clusters(vector<T> data, vector<float*> cluster_points,
        int clusters_num, int features_num) {
    vector<vector<T> > clusters;
    for (int i = 0; i < clusters_num; i++) {
        clusters.push_back(vector<T>());
    }

    for (auto p : data) {
        int closest_cl = 0;
        float closest_cl_dist = dej_dist(p, cluster_points[0], features_num);

        for (int i = 1; i < cluster_points.size(); i++) {
            float d = dej_dist(p, cluster_points[i], features_num);
            if (d < closest_cl_dist) {
                closest_cl_dist = d;
                closest_cl = i;
            }
        }

        clusters[closest_cl].push_back(p);
    }
    return clusters;
}

template <typename T>  // T is either bool* of float*
int closest_cluster(T point, vector<float*> cluster_points, int features_num) {
    int closest_cl = -1;
    int closest_cl_dist = 100;
    for (int i = 0; i < cluster_points.size(); i++) {
        float d = dej_dist(point, cluster_points[i], features_num);
        if (d < closest_cl_dist) {
            closest_cl_dist = d;
            closest_cl = i;
        }
    }
    return closest_cl;
}

template <typename T>  // T is either bool* of float*
float closest_cluster_dist(
        T point, vector<float*> cluster_points, int features_num) {
    return dej_dist(
        point,
        cluster_points[closest_cluster(point, cluster_points, features_num)],
        features_num);
}

// ===================BinaryColon specific functions

void combine_cluster_points(
        Mat &I, float* cluster_distances, vector<float*> cluster_points) {
    float cluster_distances_sum = arr_sum(
        cluster_distances, cluster_distances + cluster_points.size());
    float * p;
    for (int i = 0; i < I.rows; i++) {
        p = I.ptr<float>(i);
        for (int j = 0; j < I.cols; j++) {
            p[j] = 0;
            for (int cli = 0; cli < cluster_points.size(); cli++)
                p[j] += (
                    cluster_points[cli][i * I.cols + j] *
                    (cluster_distances[cli] / cluster_distances_sum));
            p[j] = 1.f - p[j];
        }
    }
}

template <typename T>  // T is either bool* of float*
float find_distortion(
        float* cluster_point, vector<T> cluster, int features_num) {
    float distortion = 0;
    for (T p : cluster)
        distortion += dej_dist(p, cluster_point, features_num);
    return static_cast<float>(distortion / cluster.size());
}

BinaryColon::BinaryColon(int fn, int cn) {
    features_num = fn;
    clusters_num = cn;

    capacity = clusters_num * cluster_targ_size;
    points_num = 0;

    float * point;
    cluster_sizes = new int[clusters_num];
    for (int i = 0; i != clusters_num; i++) {
        point = new float[features_num];
        for (int i = 0; i != features_num; i++)
            point[i] = rand01();
        cluster_points.push_back(point);
    }
}

void BinaryColon::feed(float* point) {
    // Analyze Activity
    DATA_TYPE* binary_point = float_to_binary(point, features_num);
    // DATA_TYPE* binary_point = new DATA_TYPE[features_num];
    // for (int i = 0; i < features_num; i++) {
    //     binary_point[i] = point[i];
    // }

    // DATA_TYPE* binary_point = reduce_dimentionality(point, features_num, 16);

    float cap_pos = min(
        static_cast<float>(capacity) / static_cast<float>(points_num), 1.f);

    if (prob(cap_pos)) {
        if (points_num >= capacity) {
            int erase_pos = rand() % data.size();
            delete [] *(data.begin() + erase_pos);
            data.erase(data.begin() + erase_pos);
        }
        data.push_back(binary_point);
        points_num++;
    }

    vector<vector<DATA_TYPE*>> clusters = get_clusters(
        data, cluster_points, clusters_num, features_num);

    for (int cli = 0; cli < clusters_num; cli++) {
        cluster_sizes[cli] = clusters[cli].size();

        float* new_cluster_points_p = mean_point_speedmod(
            clusters[cli], cluster_points[cli], features_num, 0.1);
        delete [] cluster_points[cli];
        cluster_points[cli] = new_cluster_points_p;
    }
    clusters.clear();
}

float* BinaryColon::get_cluster_distances(float* point) {
    float* ans = new float[clusters_num];
    for (int cli = 0; cli < clusters_num; cli++) {
        ans[cli] = dej_dist(
            point, cluster_points[cli], features_num) / static_cast<float>(
                features_num);
    }

    return ans;
}

// ===================Finding closest points

class Node_сoord {
 public:
    float c;
    int id;
    Node_сoord *next;
    Node_сoord *prev;
};

class Sorted_coord {
 public:
    Node_сoord* head;

    Sorted_coord() {
        head = NULL;
    }

    Node_сoord* add(float c, int id) {
        if (head == NULL) {
            head = new Node_сoord();
            head->c = c;
            head->id = id;
            head->prev = NULL;
            head->next = NULL;
            return head;
        } else {
            Node_сoord * new_Node = new Node_сoord();
            new_Node->c = c;
            new_Node->id = id;

            Node_сoord* insert_after = head;
            while (true) {
                if (insert_after->c > c) {
                    insert_after = insert_after->prev;
                    break;
                }
                if (insert_after->next != NULL)
                    insert_after = insert_after->next;
                else
                    break;
            }

            new_Node->prev = insert_after;
            if (insert_after != NULL)
                new_Node->next = insert_after->next;
            else
                new_Node->next = head;

            if (insert_after != NULL)
                insert_after->next = new_Node;
            if (new_Node->next != NULL)
                new_Node->next->prev = new_Node;

            if (insert_after == NULL)
                head = new_Node;

            return new_Node;
        }
    }
};

class Points {
 public:
    Sorted_coord* coord;

    Points() {
        coord = new Sorted_coord[3];
    }

    vector<int> add_point_get_closest(
            int id, float* xyz, float receptive_range) {
        Node_сoord ** node = new Node_сoord*[3];
        for (int i = 0; i < 3; i ++)
            node[i] = coord[i].add(xyz[i], id);

        vector<pair<int, int>> id_set;
        for (int i = 0; i < 3; i ++) {
            Node_сoord * pointer = node[i]->prev;
            while (pointer != NULL) {
                if (pointer->c < node[i]->c - receptive_range) {
                    pointer = pointer->next;
                    break;
                }
                pointer = pointer->prev;
            }
            if (pointer == NULL) {
                pointer = coord[i].head;
            }
            while (pointer != NULL) {
                if (pointer->c > node[i]->c + receptive_range) {
                    break;
                }
                bool id_found = false;
                for (auto &p : id_set) {
                    if (p.first == pointer->id) {
                        p.second += 1;
                        id_found = true;
                        break;
                    }
                }
                if (!id_found)
                    id_set.push_back(make_pair(pointer->id, 1));
                pointer = pointer->next;
            }
        }
        vector<int> result_ids;
        for (auto p : id_set)
            if (p.second == 3 && p.first != id)
                result_ids.push_back(p.first);

        delete [] node;

        return result_ids;
    }
};

// ===================train & run functions

void image_train(BinaryColon &bc, string image_location, int iterations, bool show) {
    Mat I;
    imread(image_location, IMREAD_GRAYSCALE).convertTo(I, CV_32FC1, 1. / 255.);

    int clusters_num = bc.clusters_num;
    int res = round(sqrt(bc.features_num));

    if (show)
        create_cluster_windows(clusters_num);

    int time_began = time(0);
    uint64 prev_log_time = 0;

    for (int it = 0; it <= iterations; it ++) {
        if (log_delay(&prev_log_time)) {
            print_progress(it, iterations, time_began);
            if (show)
                show_clusters(bc);
        }
        int x = rand() % (I.cols - res - 1);
        int y = rand() % (I.rows - res - 1);
        Mat P(I, Rect(x, y, res, res));

        float * fl_p = Mat_to_float(P);
        bc.feed(fl_p);
        delete [] fl_p;
    }
    cout << "\n";

    if (show) {
        cout << "Press any key on image to exit\n";
        cout.flush();
        waitKey(0);
        destroyAllWindows();
    }
}

void image_train_2_layers(string image_location, int iterations, bool show) {
    Mat I;
    imread(image_location, IMREAD_GRAYSCALE).convertTo(I, CV_32FC1, 1. / 255.);

    int res = 9;

    int time_began = time(0);
    uint64 prev_log_time = 0;

    for (int it = 0; it <= iterations; it ++) {
        if (log_delay(&prev_log_time))
            print_progress(it, iterations, time_began);

        int x = rand() % (I.cols - res - 1);
        int y = rand() % (I.rows - res - 1);

        Mat P(I, Rect(x, y, res, res));
    }
    cout << "\n";
}

void Hebb_image_train(string image_location, int iterations) {
    Mat I;
    imread(image_location, IMREAD_GRAYSCALE).convertTo(I, CV_32FC1, 1. / 255.);
}

void photo_proceed_by_bc(BinaryColon &bc, string image_location) {
    Mat I;
    imread(image_location, IMREAD_GRAYSCALE).convertTo(I, CV_32FC1, 1. / 255.);

    int clusters_num = bc.clusters_num;
    int res = sqrt(bc.features_num);

    int time_began = time(0);
    uint64 prev_log_time = 0;

    Mat* O = new Mat[res * res];
    for (int oi = 0; oi < res * res; oi ++)
        O[oi] = I.clone();

    for (int x_shift = 0; x_shift < 1; x_shift ++)
        for (int y_shift = 0; y_shift < 1; y_shift ++)
            for (
                    int x = x_shift;
                    x < O[res * x_shift + y_shift].cols - res;
                    x += res)
                for (
                        int y = y_shift;
                        y < O[res * x_shift + y_shift].rows - res;
                        y += res
                    ) {
                    if (log_delay(&prev_log_time))
                        print_progress(x * O[res * x_shift + y_shift].rows + y,
                                    (
                                        O[res * x_shift + y_shift].cols *
                                        O[res * x_shift + y_shift].rows),
                                    time_began);

                    Mat P(O[res * x_shift + y_shift], Rect(x, y, res, res));

                    DATA_TYPE* fl_p = Mat_to_float(P);
                    DATA_TYPE* binary_point = float_to_binary(fl_p, bc.features_num);

                    float * cluster_distances = bc.get_cluster_distances(binary_point);

                    combine_cluster_points(P,
                        cluster_distances, bc.cluster_points);

                    delete [] fl_p;
                    delete [] binary_point;
                    delete [] cluster_distances;
                }
    cout << "\n";

    Mat_combine(I, O, 1);

    namedWindow("Unprocessed");
    imshow("Unprocessed", imread(image_location, IMREAD_GRAYSCALE));
    namedWindow("Processed");
    imshow("Processed", I);
    cout << "Press any key on image to exit\n";
    cout.flush();
    waitKey(0);
    destroyAllWindows();
    delete [] O;
}

int main(int argc, char **argv) {
    if (argc != 3 || (!strcmp(argv[1], "--train") && !strcmp(argv[1], "--test"))) {
        cout << "Usage:\n";
        cout << "To train:\n";
        cout << argv[0] << " --train " << "image.jpg\n";
        cout << "To test:\n";
        cout << argv[0] << " --test " << "image.jpg\n";
        return 1;
    }

    srand(time(0));
    string mode = argv[1];
    string image_name = argv[2];

    if (!file_exists(image_name)) {
        cout << "Error: " << image_name << " doesn't exist!\n";
        return 1;
    }

    int res = 3, clusters_num = 10;
    bool show = true;
    int iterations = 10000;
    int settings_num = 2;
    cout << "1) res = " << res << "\n2) clusters_num = " << clusters_num << "\n";

    if (mode == "--train") {
        settings_num = 4;
        cout << "3) show = " << ((show)?("true"):("false")) << "\n";
        cout << "4) iterations = " << iterations << "\n";
    }

    while (true) {
        cout << "Do you want to change settings? (y/n): ";
        char ans;
        cin >> ans;
        if ((ans == 'y') || (ans == 'Y')) {
            int sett_num = 0;
            while (!(sett_num >= 1 && sett_num <= settings_num)) {
                cout << "Enter setting number (1.." << settings_num << "): ";
                cin >> sett_num;
            }
            if (sett_num == 1) {
                cout << "res = ";
                cin >> res;
            } else if (sett_num == 2) {
                cout << "clusters_num = ";
                cin >> clusters_num;
            } else if (sett_num == 3) {
                cout << "show = (1 or 0): ";
                cin >> show;
            } else if (sett_num == 4) {
                cout << "iterations = ";
                cin >> iterations;
            }
        } else {
            break;
        }
    }
    stringstream bc_filename;
    bc_filename << "bc_" << res * res << "_" << clusters_num << "_float.txt";

    if (mode == "--train") {
        BinaryColon bc = BinaryColon(res * res, clusters_num);

        image_train(bc, IMAGE_LOCATION_PREFIX + image_name, iterations, show);
        save_bc(bc, bc_filename.str());
    } else if (mode == "--test") {
        if (!file_exists(bc_filename.str())) {
            cout << "Error: " << bc_filename.str() << "doesn't exist!\n";
            return 1;
        }

        BinaryColon bc = read_bc(bc_filename.str());
        photo_proceed_by_bc(bc, IMAGE_LOCATION_PREFIX + image_name);
    }

    return 0;
}
