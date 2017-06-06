/* 
 * File:   Rank.hpp
 * Author: Fernando B Oliveira <fboliveira25@gmail.com>
 *
 * Created on November 6, 2014, 4:41 PM
 */

#ifndef RANK_HPP
#define	RANK_HPP

class Rank {
    
    int source;
    int id;
    float cost;
    
public:
    
    Rank();    
    Rank(int source, int id, float cost);
    Rank(const Rank& other);

    virtual ~Rank();
    
    float getCost() const;
    void setCost(float cost);

    int getId() const;
    void setId(int id);

    int getSource() const;
    void setSource(int source);

    void print();

    static bool compare(Rank i, Rank j);
    
private:

};

#endif	/* RANK_HPP */

