#pragma once

class int2 {
public:
	int x;
	int y;

	int2(int x, int y) {
		this->x = x;
		this->y = y;
	}

	int2& operator+(const int2& a){
		this->x+= a.x;
		this->y+= a.y;

		return *this;
	}
};
