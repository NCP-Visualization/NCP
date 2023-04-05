#include<vector>
#include<tuple>
#include<list>
#include<fstream>
#include<algorithm>
#include<iostream>
#include<set>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using namespace std;

constexpr auto PI = 3.141592654;
constexpr auto sqrt2 = 1.4142135623731;

// 点类
class Point
{
public:
	double x = 0;
	double y = 0;
public:
	Point(double x_in, double y_in) : x(x_in), y(y_in)
	{}

	Point(const Point& p)
	{
		x = p.x;
		y = p.y;
	}

	Point()
	{}

	~Point()
	{}

	bool operator==(const Point& other) const
	{
		return (other.x == x && other.y == y);
	}
};

// 双向链表
class LinkedListNode
{
public:
	LinkedListNode* next;
	LinkedListNode* previous;
	int index;
	tuple<Point, double> data;

public:
	LinkedListNode()
	{
		next = nullptr;
		previous = nullptr;
		index = -1;
	}

	LinkedListNode(int _index, tuple<Point, double> _data)
	{
		next = nullptr;
		previous = nullptr;
		index = _index;
		data = _data;
	}
};


class D3SimiFrontChain
{
public:
	vector<double> cir;
	vector<vector<double>> similarity;
	vector<vector<double>> pos;
	vector<double> r;
	vector<int> index;
	vector<int> index_inverse;

	int stop_iter = -1;
	// debug

public:
	// 输出vector<tuple<Point, double>>表示packing结果的圆心坐标Point以及圆半径double（如果是正方形，则是半边长）
	vector<tuple<Point, double>> getCirclePacking()
	{
		int n = cir.size();
		if (n == 0)
			return {};

		vector<tuple<Point, double>> res(n);

		// put the first circle
		auto first = cir[index[0]];
		LinkedListNode* a = new LinkedListNode(index[0], tuple<Point, double>(Point(0, 0), first));
		res[index[0]] = a->data;
		if (n == 1)
		{
			delete a;
			return res;
		}

		// put the second circle
		set<int> candidates;
		for (int i = 1; i < index.size(); i++)
		{
			candidates.insert(i);
		}
		int second_index = 0;
		double second_max_simi = 0;
		for (auto it = candidates.begin(); it != candidates.end(); it++)
		{
			if (similarity[index[0]][index[*it]] > second_max_simi)
			{
				second_max_simi = similarity[index[0]][index[*it]];
				second_index = *it;
			}
		}
		// select the second circle
		auto second = cir[index[second_index]];
		delete a;
		a = new LinkedListNode(index[0], tuple<Point, double>(Point(-second, 0), first));
		LinkedListNode* b = new LinkedListNode(index[second_index], tuple<Point, double>(Point(first, 0), second));
		res[index[0]] = a->data;
		res[index[second_index]] = b->data;
		if (n == 2)
		{
			delete a;
			delete b;
			return res;
		}
		candidates.erase(second_index);

		// put the third circle
		int third_index = 0;
		double third_max_simi = 0;
		for (auto it = candidates.begin(); it != candidates.end(); it++)
		{
			if (similarity[index[0]][index[*it]] + similarity[index[second_index]][index[*it]] > third_max_simi)
			{
				third_max_simi = similarity[index[0]][index[*it]] + similarity[index[second_index]][index[*it]];
				third_index = *it;
			}
		}
		auto third = cir[index[third_index]];
		auto pos = place(b->data, a->data, third);
		LinkedListNode* c = new LinkedListNode(index[third_index], pos);
		res[index[third_index]] = c->data;
		candidates.erase(third_index);

		// build list  
		a->next = b;
		c->previous = b;
		b->next = c;
		a->previous = c;
		c->next = a;
		b->previous = a;

		LinkedListNode* chain = a;

		int i = 0;
		int num = 0;
		while (!candidates.empty())
		{
			if (num == stop_iter - 3)
				break;
			double simi_max = 0;
			int selected_index = 0;
			tuple<Point, double> selected;
			LinkedListNode* before = nullptr;
			LinkedListNode* after = nullptr;

			for (auto it = candidates.begin(); it != candidates.end(); it++)
			{
				auto experiment = tryToPutCircle(a, a->next, cir[index[*it]]);
				int index_before = get<1>(experiment)->index;
				int index_after = get<2>(experiment)->index;
				double simi_two = similarity[index_before][index[*it]] + similarity[index[*it]][index_after];
				if (simi_max < simi_two)
				{
					simi_max = simi_two;
					selected = get<0>(experiment);
					before = get<1>(experiment);
					after = get<2>(experiment);
					selected_index = *it;
				}
			}

			c = new LinkedListNode(index[selected_index], selected);
			res[index[selected_index]] = selected;

			// build chain
			LinkedListNode* p = before->next;
			while (p != after)
			{
				if (p == chain)
					chain = chain->next;
				LinkedListNode* temp = p->next;
				delete p;
				p = temp;
			}
			before->next = c;
			c->previous = before;
			c->next = after;
			after->previous = c;
			candidates.erase(selected_index);

			// compute the new closest circle pair to the centroid
			a = before;
			double aa = score(a);
			while (c != after)
			{
				double ca = score(c);
				if (ca < aa)
				{
					a = c;
					aa = ca;
				}
				c = c->next;
			}
			num++;
		}

		// delete list
		chain = destroy(chain);

		return res;
	}

	tuple<Point, double> place(tuple<Point, double> cir1, tuple<Point, double> cir2, double r)
	{
		double ax = get<0>(cir1).x;
		double ay = get<0>(cir1).y;
		double ar = get<1>(cir1);
		double bx = get<0>(cir2).x;
		double by = get<0>(cir2).y;
		double br = get<1>(cir2);

		double da = br + r;
		double db = ar + r;

		double a = ax, b = ay, c = db;
		double d = bx, e = by, f = da;
		double p = 2 * d - 2 * a, q = 2 * e - 2 * b, k = a * a - d * d + b * b - e * e + f * f - c * c;
		double t, s;
		double A, B, C;
		double delta;
		double x1, x2, y1, y2;

		if (q == 0.0)
		{
			x1 = -k / p;
			x2 = -k / p;
			t = c * c - (x1 - a) * (x1 - a);
			if (t < 0)
				t = 0;
			y1 = b + sqrt(t);
			y2 = b - sqrt(t);
		}
		else
		{
			t = -p / q;
			s = -k / q;
			A = 1 + t * t;
			B = -2 * a + 2 * t * (s - b);
			C = a * a + (s - b) * (s - b) - c * c;
			delta = B * B - 4 * A * C;
			if (delta < 0)
				delta = 0;
			x1 = (-B + sqrt(delta)) / (2 * A);
			x2 = (-B - sqrt(delta)) / (2 * A);
			y1 = t * x1 + s;
			y2 = t * x2 + s;
		}
		double mul = (b - e) * (x1 - d) - (a - d) * (y1 - e);
		if (mul < 0)
			return tuple<Point, double>(Point(x1, y1), r);
		return tuple<Point, double>(Point(x2, y2), r);
	}

	bool intersects(tuple<Point, double> cir1, tuple<Point, double> cir2)
	{
		double dx = get<0>(cir2).x - get<0>(cir1).x;
		double dy = get<0>(cir2).y - get<0>(cir1).y;
		double dr = get<1>(cir2) + get<1>(cir1);
		return (dr * dr - 1e-6 > dx * dx + dy * dy);
	}

	double score(LinkedListNode* node)
	{
		auto a = node->data;
		auto b = node->next->data;
		double ab = get<1>(a) + get<1>(b);

		double ax = get<0>(a).x;
		double ay = get<0>(a).y;
		double ar = get<1>(a);

		double bx = get<0>(b).x;
		double by = get<0>(b).y;
		double br = get<1>(b);

		double dx = (ax * br + bx * ar) / ab;
		double dy = (ay * br + by * ar) / ab;
		return dx * dx + dy * dy;
	}

	tuple<tuple<Point, double>, LinkedListNode*, LinkedListNode*> tryToPutCircle(LinkedListNode* a, LinkedListNode* b, double r)
	{
		auto pos = place(a->data, b->data, r);
		LinkedListNode* p = b;
		LinkedListNode* q = a;
		while (true)
		{
			p = p->next;
			if (p == q)
				break;
			if (intersects(pos, p->data))
				//新圆的放置位置与FrontChain中的其它圆相交，递归求解
				return tryToPutCircle(a, p, r);

			q = q->previous;
			if (p == q)
				break;
			if (intersects(pos, q->data))
				//新圆的放置位置与FrontChain中的其它圆相交，递归求解
				return tryToPutCircle(q, b, r);
		}
		//新圆的放置位置与FrontChain中的其它圆都不相交，此位置为输出结果
		return tuple<tuple<Point, double>, LinkedListNode*, LinkedListNode*>(pos, a, b);
	}

	LinkedListNode* destroy(LinkedListNode* head)
	{
		LinkedListNode* abandoned = nullptr;
		while (head->next != nullptr)
		{
			if (head->previous)
				head->previous->next = nullptr;
			abandoned = head->next;
			abandoned->previous = nullptr;
			delete head;
			head = abandoned;
		}
		if (head)
			delete head;
		head = nullptr;
		abandoned = nullptr;
		return head;
	}

	void set_importance(const vector<double>& importance)
	{
		cir = importance;
	}

	void set_similarity(const vector<vector<double>>& simi)
	{
		similarity = simi;
	}

	void set_index(const vector<int>& _index)
	{
		index = _index;
	}

	void set_stop_iter(const int& it)
	{
		stop_iter = it;
	}

	void layout()
	{
		auto res = getCirclePacking();
		pos = vector<vector<double>>(res.size(), vector<double>(2, 0));
		r = vector<double>(res.size(), 0);

		double minx = 1e8;
		double miny = 1e8;
		double maxx = 0;
		double maxy = 0;
		for (auto it : res)
		{
			minx = min(minx, get<0>(it).x - get<1>(it));
			miny = min(miny, get<0>(it).y - get<1>(it));
			maxx = max(maxx, get<0>(it).x + get<1>(it));
			maxy = max(maxy, get<0>(it).y + get<1>(it));
		}
		double min_v = min(minx, miny);
		double scale = max(maxx - minx, maxy - miny);

		for (int i = 0; i < res.size(); i++)
		{
			auto it = res[i];
			pos[i] = { (get<0>(it).x - minx) / scale, (get<0>(it).y - miny) / scale };
			r[i] = get<1>(it) / scale;
		}
	}

	vector<vector<double>> get_pos()
	{
		return pos;
	}

	vector<double> get_r()
	{
		return r;
	}

	void print(tuple<Point, double> pos)
	{
		cout << get<0>(pos).x << ' ' << get<0>(pos).y << ' ' << get<1>(pos) << endl;
	}
};

PYBIND11_MODULE(D3SimiFrontChain, m) {
	m.doc() = "pybind11 example";
	pybind11::class_<D3SimiFrontChain>(m, "D3SimiFrontChain")
		.def(pybind11::init())
		.def("set_importance", &D3SimiFrontChain::set_importance)
		.def("layout", &D3SimiFrontChain::layout)
		.def("get_r", &D3SimiFrontChain::get_r)
		.def("get_pos", &D3SimiFrontChain::get_pos)
		.def("set_index", &D3SimiFrontChain::set_index)
		.def("set_stop_iter", &D3SimiFrontChain::set_stop_iter)
		.def("set_similarity", &D3SimiFrontChain::set_similarity);
}

//int main()
//{
//    FrontChain f;
//    f.layout();
//}
