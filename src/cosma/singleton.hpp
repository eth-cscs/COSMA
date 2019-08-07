#pragma once
class singleton {
public:
    static singleton& get_instance() {
        static singleton instance;
        return instance;
      }

private:
    singleton()= default;
    ~singleton()= default;
    // delete these to be sure it's not copied
    singleton(const singleton&)= delete;
    singleton& operator=(const singleton&)= delete;
};
